"""Probability map -> disjoint filament masks (Arch section 5.2, decisions D4, D7, D8, D10).

Steps, in the order of A5.2 (every parameter comes from :class:`PostprocessConfig`, i.e. from the config file):

1. upsample the (averaged) probability map bilinearly to 2048 x 2048 (probabilities, not masks: ceiling 0.977 vs 0.894);
2. zero everything outside ``r_max`` of the disc radius (optional; needs the disc, ``disc.py``);
3. threshold at ``threshold`` (strictly greater), or hysteresis: keep pixels above ``t_low`` that are 8-connected to a
   pixel above ``t_high`` (optional; hypothesis, untested);
4. morphological closing with a disk of radius ``closing_radius`` px (optional);
5. 8-connected components (D7); fill the holes of each component;
6. resolve overlaps created by the filling: a pixel claimed by two components stays with the larger (filled) one;
7. drop components with area < ``a_min``, mean probability < ``c_min`` or centroid r/R > ``c_limb`` (D8);
8. number the survivors 1..n in raster order of their first pixel; the caller makes ``<image>_<k>`` ids.

Defaults reproduce the model-comparison setting of A6: no filters, no disc masking, only the threshold.

Why the output is disjoint by construction: a *label map* holds one id per pixel, and masks are cut from it.  The CSV
checker (rle.py) still verifies this on the decoded masks before any upload.

Two derived facts used in step 6 (explained, not assumed): with 8-connected foreground, a hole is a background region
that is 4-connected and does not reach the image border (``scipy.ndimage.binary_fill_holes`` uses exactly this
background connectivity); and a component that touches a hole of another component lies entirely inside that hole,
so overlaps after filling are always nestings and the smaller component simply disappears.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from scipy import ndimage

from .disc import Disc, radial_map
from .rle import encode_crop, encode_crop_rle, encode_mask  # noqa: F401

FULL = 2048


@dataclass(frozen=True)
class PostprocessConfig:
    threshold: float = 0.5  # t; A5.3 grid 0.20 .. 0.55, comparison arms 0.30 / 0.40 / 0.50 (A6)
    hysteresis: tuple[float, float] | None = None  # (t_low, t_high); replaces ``threshold`` when set
    closing_radius: int = 0  # rho in px on the 2048 grid; 0 = off (A5.3: 0, 1, 2, 4)
    fill_holes: bool = True
    r_max: float | None = None  # zero outside r/R > r_max; None = no disc masking (A5.3: 0.97, 1.00, none)
    a_min: int = 0  # minimum area on the 2048 grid; 0 = off
    c_min: float = 0.0  # minimum mean probability; 0 = off
    c_limb: float | None = None  # maximum centroid r/R; None = off
    out_size: int | None = FULL  # upsample target; None keeps the input size (used by unit tests)

    @classmethod
    def from_dict(cls, d: dict | None) -> "PostprocessConfig":
        d = dict(d or {})
        if d.get("hysteresis") is not None:
            d["hysteresis"] = tuple(d["hysteresis"])
        return cls(**d)

    def needs_disc(self) -> bool:
        return self.r_max is not None or self.c_limb is not None


@dataclass
class Component:
    """One connected component after hole filling and overlap resolution (before filtering)."""

    orig_label: int  # label from the 8-connected component search (raster order)
    area: int  # pixels, on the output grid
    mean_prob: float
    max_prob: float
    r_over_R: float  # centroid distance / disc radius; nan when no disc was given
    bbox: tuple[int, int, int, int]  # y0, x0, y1, x1 (half-open)
    kept: bool = True
    drop_reason: str | None = None  # "absorbed" | "a_min" | "c_min" | "c_limb"
    new_id: int = 0  # 1..n for kept components, 0 otherwise


@dataclass
class PostprocessResult:
    labels: np.ndarray  # (H, W) int32; 0 background, 1..n kept components
    components: list[Component] = field(default_factory=list)

    @property
    def kept(self) -> list[Component]:
        return sorted((c for c in self.components if c.kept), key=lambda c: c.new_id)

    @property
    def n_kept(self) -> int:
        return sum(c.kept for c in self.components)

    def mask(self, new_id: int) -> np.ndarray:
        """(H, W) uint8 mask of kept component ``new_id`` (1-based)."""
        c = next(c for c in self.components if c.kept and c.new_id == new_id)
        y0, x0, y1, x1 = c.bbox
        m = np.zeros(self.labels.shape, np.uint8)
        m[y0:y1, x0:x1] = self.labels[y0:y1, x0:x1] == new_id
        return m

    def _crop(self, c: "Component") -> tuple[np.ndarray, int, int]:
        y0, x0, y1, x1 = c.bbox
        return self.labels[y0:y1, x0:x1] == c.new_id, y0, x0

    def rles(self) -> list[dict]:
        """pycocotools RLEs of the kept components, in id order (input of the fast PQ scorer).

        Built from each component's bounding-box crop: the cost does not grow with 2048 x 2048 per component."""
        h, w = self.labels.shape
        return [encode_crop_rle(*self._crop(c), (h, w)) for c in self.kept]

    def counts(self) -> list[str]:
        """Counts strings for the CSV, in id order (same strings as ``rle.encode_mask`` of the full masks)."""
        h, w = self.labels.shape
        return [encode_crop(*self._crop(c), (h, w)) for c in self.kept]


def upsample_probability(prob: np.ndarray, out_size: int | None = FULL) -> np.ndarray:
    """Bilinear upsample (half-pixel-centre convention, same as torch ``align_corners=False``) to float32."""
    p = np.asarray(prob, dtype=np.float32)
    if out_size is None or p.shape == (out_size, out_size):
        return p
    return cv2.resize(p, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def disk_kernel(radius: int) -> np.ndarray:
    """Discrete disk of radius ``radius`` px: all offsets with dx^2 + dy^2 <= radius^2."""
    r = int(radius)
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return (xx * xx + yy * yy <= r * r).astype(np.uint8)


def _hysteresis(p: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    """Pixels above t_low in 8-connected components that contain a pixel above t_high."""
    if t_low > t_high:
        raise ValueError("t_low %.3f is above t_high %.3f" % (t_low, t_high))
    low = (p > t_low).astype(np.uint8)
    n, lab = cv2.connectedComponents(low, connectivity=8)
    if n < 2:
        return np.zeros(p.shape, bool)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[p > t_high])] = True
    keep[0] = False
    return keep[lab]


def binarise(p: np.ndarray, cfg: PostprocessConfig) -> np.ndarray:
    """Steps 3 and 4: threshold (or hysteresis), then optional closing.  Returns bool."""
    if cfg.hysteresis is not None:
        m = _hysteresis(p, cfg.hysteresis[0], cfg.hysteresis[1])
    else:
        m = p > cfg.threshold
    if cfg.closing_radius > 0:
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, disk_kernel(cfg.closing_radius)).astype(bool)
    return m


def postprocess(
    prob: np.ndarray,
    cfg: PostprocessConfig = PostprocessConfig(),
    disc: Disc | None = None,
    radial: np.ndarray | None = None,
) -> PostprocessResult:
    """Run steps 1 to 8 on one probability map (any resolution; upsampled to ``cfg.out_size``).

    ``disc`` is in the coordinates of the *output* grid.  ``radial`` may be a precomputed ``radial_map`` (saves time
    when many settings are scored on one image).
    """
    p = upsample_probability(prob, cfg.out_size)
    h, w = p.shape
    if cfg.needs_disc():
        if disc is None:
            raise ValueError("r_max / c_limb need the disc (disc.fit_disc) but none was given")
        if radial is None:
            radial = radial_map(p.shape, disc)
    if cfg.r_max is not None:
        p = np.where(radial > cfg.r_max, np.float32(0.0), p)

    binary = binarise(p, cfg)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)

    # step 5/6: fill the holes of each component, claim pixels from the largest filled component downwards
    crops = []  # (orig_label, y0, x0, y1, x1, filled crop, original-component crop)
    for k in range(1, n):
        x, y, bw, bh, _ = stats[k]
        y0, x0 = max(y - 1, 0), max(x - 1, 0)
        y1, x1 = min(y + bh + 1, h), min(x + bw + 1, w)
        comp = lab[y0:y1, x0:x1] == k
        filled = ndimage.binary_fill_holes(comp) if cfg.fill_holes else comp
        crops.append((k, y0, x0, y1, x1, filled, comp))
    crops.sort(key=lambda c: (-int(c[5].sum()), c[0]))  # largest filled area first; ties by label

    out = np.zeros((h, w), np.int32)
    comps: list[Component] = []
    for k, y0, x0, y1, x1, filled, comp in crops:
        view = out[y0:y1, x0:x1]
        region = filled & (view == 0)
        area = int(region.sum())
        if area == 0:
            comps.append(Component(k, 0, 0.0, 0.0, float("nan"), (y0, x0, y1, x1), False, "absorbed"))
            continue
        view[region] = k
        pv = p[y0:y1, x0:x1][region]
        ys, xs = np.nonzero(region)
        if disc is not None:
            cx, cy = x0 + xs.mean() + 0.5, y0 + ys.mean() + 0.5
            r_over_R = float(np.hypot(cx - disc.cx, cy - disc.cy) / disc.radius)
        else:
            r_over_R = float("nan")
        comps.append(Component(k, area, float(pv.mean()), float(pv.max()), r_over_R, (y0, x0, y1, x1)))

    # step 7: filters
    for c in comps:
        if not c.kept:
            continue
        if c.area < cfg.a_min:
            c.kept, c.drop_reason = False, "a_min"
        elif c.mean_prob < cfg.c_min:
            c.kept, c.drop_reason = False, "c_min"
        elif cfg.c_limb is not None and c.r_over_R > cfg.c_limb:
            c.kept, c.drop_reason = False, "c_limb"

    # step 8: number the survivors in raster order of their label
    comps.sort(key=lambda c: c.orig_label)
    lut = np.zeros(n + 1, np.int32)
    new_id = 0
    for c in comps:
        if c.kept:
            new_id += 1
            c.new_id = new_id
            lut[c.orig_label] = new_id
    return PostprocessResult(labels=lut[out], components=comps)
