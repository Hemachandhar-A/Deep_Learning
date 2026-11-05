"""Solar-disc detection (Arch section 3, "Disc detection recipe"; used by P3, P4 and the limb rule of section 5).

The recipe, implemented exactly as in the analysis scripts (radial2.py, limb_profile.py) that produced the measured
numbers (radius median 891.4 px over the 707 training images):

1. downscale the image to 512 px by area interpolation;
2. Otsu threshold; keep the largest 8-connected component;
3. threshold again at 0.65 x the median gray level inside that component; keep the largest 8-connected component
   (the *bright disc*; Otsu alone gives about 904 px, which is too large);
4. centre = centroid of the bright disc, radius = sqrt(area / pi); multiply both by 4 for the 2048 px grid.

Coordinates: continuous, with pixel (i, j) covering [i, i+1) x [j, j+1), so a pixel centre is (i + 0.5, j + 0.5).  The
512 px centroid is therefore ``mean_index + 0.5`` before multiplying by 4, as in radial2.py.  The radius is the same in
either convention.

Used only to *post-process* (zero outside the disc, limb rule) and for the optional limb-flattening arm; the network
input is never masked or cropped (P3).  The fit has not yet been checked visually on Bh and Th images (A3);
``scripts/overlay_check.py`` writes the overlays for that check.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

RECIPE_SIZE = 512  # step 1
BRIGHT_FRACTION = 0.65  # step 3


@dataclass(frozen=True)
class Disc:
    """Bright disc in the coordinates of the input image (default 2048 x 2048)."""

    cx: float
    cy: float
    radius: float


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n < 2:
        raise ValueError("no foreground component found: the image has no bright disc")
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == k


def fit_disc(image_u8: np.ndarray) -> Disc:
    """Fit the bright disc of one 8-bit grayscale image (the recipe above).  Input side must be a multiple of 512."""
    if image_u8.ndim != 2 or image_u8.dtype != np.uint8:
        raise ValueError("expected a 2-D uint8 image, got %s %s" % (image_u8.shape, image_u8.dtype))
    h, w = image_u8.shape
    if h != w or h % RECIPE_SIZE:
        raise ValueError("image side must be a square multiple of %d, got %s" % (RECIPE_SIZE, image_u8.shape))
    scale = h / RECIPE_SIZE
    small = cv2.resize(image_u8, (RECIPE_SIZE, RECIPE_SIZE), interpolation=cv2.INTER_AREA)
    _, otsu = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    d0 = _largest_component(otsu)
    thr = BRIGHT_FRACTION * float(np.median(small[d0]))
    d1 = _largest_component(small > thr)
    ys, xs = np.nonzero(d1)
    return Disc(
        cx=float((xs.mean() + 0.5) * scale),
        cy=float((ys.mean() + 0.5) * scale),
        radius=float(np.sqrt(d1.sum() * scale * scale / np.pi)),
    )


def radial_map(shape: tuple[int, int], disc: Disc) -> np.ndarray:
    """r / R of every pixel centre, float32, for an image of ``shape`` in the disc's coordinates."""
    h, w = shape
    yy = (np.arange(h, dtype=np.float32) + 0.5)[:, None] - np.float32(disc.cy)
    xx = (np.arange(w, dtype=np.float32) + 0.5)[None, :] - np.float32(disc.cx)
    return np.sqrt(yy * yy + xx * xx) / np.float32(disc.radius)


def scale_disc(disc: Disc, factor: float) -> Disc:
    """Express a disc on a grid ``factor`` times larger (e.g. 2 for 1024 -> 2048)."""
    return Disc(disc.cx * factor, disc.cy * factor, disc.radius * factor)


def draw_overlay(image_u8: np.ndarray, disc: Disc, out_side: int = 1024) -> np.ndarray:
    """RGB image (``out_side`` px) with the fitted circle (green, 2 px) and centre cross (red) for visual checks."""
    f = out_side / image_u8.shape[0]
    small = cv2.resize(image_u8, (out_side, out_side), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
    c = (int(round(disc.cx * f)), int(round(disc.cy * f)))
    cv2.circle(rgb, c, int(round(disc.radius * f)), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.drawMarker(rgb, c, (255, 0, 0), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
    return rgb
