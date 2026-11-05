"""Data handling: annotations, rasterisation, soft targets, cache and datasets.

Implements Arch section 3 (P1, P2, P7, P8), section 4.8 items 1-3 and the E6 augmentation options of section 4.4.

What lives here
---------------
* ``find_data_layout``     locate the JSON and the image folders by searching, never by a hard-coded path (A4.8 item 3).
* ``load_annotations``     group the 1,154 annotator versions by image name (S3).
* ``rasterise_annotation`` polygon -> mask with **pycocotools only** (S5: OpenCV polygon filling is slightly too thick).
* ``soft_target``          mean of the binary masks of an image's versions (P7, decision D3).
* ``build_cache``          1024 px uint8 images and targets as .npy files (P2, P8).
* ``FilamentDataset`` + ``SpecSampler``  training samples; every random choice is drawn by the sampler
  from one seeded generator, so a batch is a pure function of (seed, step) and tests can check it.

Conventions: arrays are (H, W); "1024 grid" means the 2 x 2 area average of the 2048 x 2048 original (P2).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

from .io_utils import atomic_write_json, finalise_file, sha256_file

TRAIN_JSON_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
FULL_SIZE = 2048
CACHE_SIZE = 1024  # P2: the network works on the 1024 grid; 512 px is derived from it (A4.8 item 1)
INSTRUMENT_CODES = ("Bh", "Ch", "Lh", "Mh", "Th", "Uh")


# --------------------------------------------------------------------------- locating files
@dataclass(frozen=True)
class DataLayout:
    """Paths of the competition files.  Nothing here is ever written to."""

    train_json: Path
    train_images: Path
    test_images: Path | None


def find_data_layout(data_root: str | Path) -> DataLayout:
    """Find the files below ``data_root`` by searching for the annotation JSON file name (A4.8 item 3)."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError("data_root does not exist: %s" % root)
    hits = sorted(root.rglob(TRAIN_JSON_NAME))
    if not hits:
        raise FileNotFoundError("%s not found below %s" % (TRAIN_JSON_NAME, root))
    if len(hits) > 1:
        raise RuntimeError("several copies of %s below %s: %s" % (TRAIN_JSON_NAME, root, hits))
    train_json = hits[0]
    train_images = train_json.parent / "train_images"
    if not train_images.is_dir():
        cands = [p for p in root.rglob("train_images") if p.is_dir()]
        if len(cands) != 1:
            raise FileNotFoundError("cannot find a unique train_images folder below %s" % root)
        train_images = cands[0]
    cands = [p for p in root.rglob("test_images") if p.is_dir()]
    test_images = cands[0] if len(cands) == 1 else None
    return DataLayout(train_json=train_json, train_images=train_images, test_images=test_images)


def list_image_names(folder: str | Path) -> list[str]:
    """Sorted image names (file name without extension) of one image folder."""
    return sorted(p.stem for p in Path(folder).iterdir() if p.suffix.lower() in (".jpeg", ".jpg"))


def image_path(folder: str | Path, name: str) -> Path:
    for ext in (".jpeg", ".jpg"):
        p = Path(folder) / (name + ext)
        if p.exists():
            return p
    raise FileNotFoundError("no image %s in %s" % (name, folder))


def instrument_of(image_name: str) -> str:
    """Two-letter site code at the end of the file name, e.g. ``20150125172714Mh`` -> ``Mh`` (A4.1)."""
    return image_name[14:]


def select_subset(names: Sequence[str], n: int, seed: int = 0) -> list[str]:
    """``n`` names spread over the instruments (round robin over sites, seeded shuffle inside a site), sorted.

    For dry runs on a small subset of images; deterministic for a seed."""
    rng = np.random.default_rng(seed)
    by: dict[str, list[str]] = {}
    for nm in sorted(names):
        by.setdefault(instrument_of(nm), []).append(nm)
    for k in by:
        by[k] = [str(x) for x in rng.permutation(by[k])]
    out: list[str] = []
    keys = sorted(by)
    while len(out) < min(n, len(names)):
        for k in keys:
            if by[k] and len(out) < n:
                out.append(by[k].pop())
    return sorted(out)


def read_gray(path: str | Path) -> np.ndarray:
    """Read an 8-bit grayscale image as (H, W) uint8.  ``imdecode`` avoids OpenCV's trouble with non-ASCII paths."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError("cannot decode image %s" % path)
    return img


# --------------------------------------------------------------------------- annotations
@dataclass
class Annotations:
    """The training annotations grouped the way the scorer needs them (S3, S4).

    ``version_id`` is the JSON image ``id`` = ``<annotatorID>-<imageName>``; each version is scored separately.
    """

    image_names: list[str]  # sorted unique image names
    versions_of: dict[str, list[str]]  # image name -> version ids, in JSON order
    image_of: dict[str, str]  # version id -> image name
    version_anns: dict[str, list[dict]]  # version id -> annotation dicts, in JSON order

    @property
    def n_versions(self) -> int:
        return len(self.image_of)

    @property
    def n_filaments(self) -> int:
        return sum(len(v) for v in self.version_anns.values())


def load_annotations(json_path: str | Path) -> Annotations:
    with Path(json_path).open("r", encoding="utf-8") as f:
        J = json.load(f)
    image_of: dict[str, str] = {}
    versions_of: dict[str, list[str]] = {}
    for rec in J["images"]:
        vid = rec["id"]
        name = os.path.splitext(rec["file_name"])[0]
        # id = <annotator>-<imageName>; the part after the first hyphen must be the image name (checked, not assumed)
        if vid.split("-", 1)[1] != name:
            raise ValueError("id %r does not end with the file stem %r" % (vid, name))
        if vid in image_of:
            raise ValueError("duplicate version id %s" % vid)
        image_of[vid] = name
        versions_of.setdefault(name, []).append(vid)
    version_anns: dict[str, list[dict]] = {vid: [] for vid in image_of}
    for a in J["annotations"]:
        version_anns[a["image_id"]].append(a)
    return Annotations(
        image_names=sorted(versions_of), versions_of=versions_of, image_of=image_of, version_anns=version_anns
    )


# --------------------------------------------------------------------------- rasterisation (pycocotools only)
def annotation_rle(ann: dict, size: int = FULL_SIZE) -> dict:
    """Polygon(s) of one filament -> one RLE at ``size`` x ``size`` (the host's own library; S5)."""
    return mask_utils.merge(mask_utils.frPyObjects(ann["segmentation"], size, size))


def rasterise_annotation(ann: dict, size: int = FULL_SIZE) -> np.ndarray:
    """(size, size) uint8 0/1 mask of one filament."""
    return mask_utils.decode(annotation_rle(ann, size))


def version_rles(anns: Sequence[dict], size: int = FULL_SIZE) -> list[dict]:
    """RLEs of all filaments of one annotator version, in JSON order."""
    return [annotation_rle(a, size) for a in anns]


def version_union_mask(rles: Sequence[dict]) -> np.ndarray:
    """Binary mask of one version: the union of its filaments (S5 counts 5,041 overlapping pixels in 5 versions)."""
    return mask_utils.decode(mask_utils.merge(list(rles)))


def soft_target_levels(per_version_masks: Sequence[np.ndarray]) -> np.ndarray:
    """Soft target = mean over the image's versions of their binary masks (P7, D3).  Float32 in [0, 1]."""
    k = len(per_version_masks)
    acc = np.zeros(per_version_masks[0].shape, np.uint8)
    for m in per_version_masks:
        acc += m
    return acc.astype(np.float32) / k


def soft_target(annotations: Annotations, image_name: str, size: int = FULL_SIZE) -> np.ndarray:
    """Soft target of one training image at 2048 px: levels 0, 1/3, 1/2, 2/3, 1 (A P7)."""
    unions = [
        version_union_mask(version_rles(annotations.version_anns[v], size)) for v in annotations.versions_of[image_name]
    ]
    return soft_target_levels(unions)


# --------------------------------------------------------------------------- resolution and normalisation
def area_downscale2(x: np.ndarray) -> np.ndarray:
    """2 x 2 area average of an (H, W) array.

    uint8 in -> uint8 out with rounding ``(sum + 2) >> 2``; this equals ``cv2.resize(..., INTER_AREA)`` at factor 2
    (verified on real images, see tests).  Float in -> float out.
    """
    h, w = x.shape
    if h % 2 or w % 2:
        raise ValueError("odd shape %s" % (x.shape,))
    if x.dtype == np.uint8:
        s = x.astype(np.uint16).reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3))
        return ((s + 2) >> 2).astype(np.uint8)
    return x.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)).astype(x.dtype)


def target_to_uint8(t: np.ndarray) -> np.ndarray:
    """Float target in [0, 1] -> uint8 by ``round(255 t)``.  Quantisation error <= 0.5/255 = 0.002 (derived)."""
    return np.rint(np.clip(t, 0.0, 1.0) * 255.0).astype(np.uint8)


def normalise(img_u8: np.ndarray, mean: float, std: float) -> np.ndarray:
    """P1: divide by 255, subtract the training mean, divide by the training standard deviation."""
    return (img_u8.astype(np.float32) / 255.0 - mean) / std


# --------------------------------------------------------------------------- cache (P8)
@dataclass
class Cache:
    """The 1024 px cache of images and targets (A P8).  Arrays are memory-mapped; nothing is copied."""

    train_names: list[str]
    test_names: list[str]
    n_versions: list[int]
    images_1024: np.ndarray  # (N, 1024, 1024) uint8
    targets_1024: np.ndarray  # (N, 1024, 1024) uint8, soft target * 255
    test_images_1024: np.ndarray | None
    meta: dict = field(default_factory=dict)

    def arm(self, resolution: int) -> tuple[np.ndarray, np.ndarray]:
        """(images, targets) uint8 at the arm's resolution: 1024 as cached, 512 by a further 2 x 2 average (A4.8 item 1).

        The 512 px arrays are built in memory (about 370 MB for the 707 images and targets): fine locally, note it for
        Kaggle's 29 GB RAM budget (Gate 1 review F5).
        """
        if resolution == CACHE_SIZE:
            return self.images_1024, self.targets_1024
        if resolution == CACHE_SIZE // 2:
            return (
                np.stack([area_downscale2(a) for a in self.images_1024]),
                np.stack([area_downscale2(a) for a in self.targets_1024]),
            )
        raise ValueError("resolution must be 1024 or 512, got %s" % resolution)

    def test_arm(self, resolution: int) -> np.ndarray:
        if self.test_images_1024 is None:
            raise ValueError("cache has no test images")
        if resolution == CACHE_SIZE:
            return self.test_images_1024
        if resolution == CACHE_SIZE // 2:
            return np.stack([area_downscale2(a) for a in self.test_images_1024])
        raise ValueError("resolution must be 1024 or 512, got %s" % resolution)


_CACHE_FILES = {
    "images": "train_images_1024.npy",
    "targets": "train_targets_1024.npy",
    "test": "test_images_1024.npy",
    "index": "cache_index.json",
}


def build_cache(
    layout: DataLayout,
    annotations: Annotations,
    cache_dir: str | Path,
    log=print,
    limit: int | None = None,
    names: list[str] | None = None,
    test_names: list[str] | None = None,
) -> dict:
    """Decode once and save 1024 px images and targets (P8); also the test images.

    Returns the index dict (also written as ``cache_index.json``), which includes the training-set mean and
    standard deviation at 1024 px (P1), the build time (A4.8 item 2: "Measure the build time first") and the SHA-256 of
    every array file, so that a cache damaged by a crash (risk K15) can be detected (``verify_cache``).

    ``limit`` builds only the first ``limit`` training images, ``names`` / ``test_names`` an explicit selection: both
    are for dry runs.  Files are written under a temporary name, fsynced and renamed; the index is written last, so a
    cache without an index is incomplete.
    """
    t0 = time.time()
    out = Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    if names is None:
        names = annotations.image_names if limit is None else annotations.image_names[:limit]
    names = list(names)
    unknown = [nm for nm in names if nm not in annotations.versions_of]
    if unknown:
        raise KeyError("images without annotations: %s" % unknown[:5])
    n = len(names)
    tmp = {k: out / (v + ".building") for k, v in _CACHE_FILES.items()}
    imgs = np.lib.format.open_memmap(tmp["images"], mode="w+", dtype=np.uint8, shape=(n, 1024, 1024))
    tgts = np.lib.format.open_memmap(tmp["targets"], mode="w+", dtype=np.uint8, shape=(n, 1024, 1024))
    s = s2 = 0.0
    npx = 0
    hist2048 = {}  # (k versions, count) -> pixels, for the P7 level histogram at 2048 px
    for i, name in enumerate(names):
        g = read_gray(image_path(layout.train_images, name))
        if g.shape != (FULL_SIZE, FULL_SIZE):
            raise ValueError("%s has shape %s" % (name, g.shape))
        small = area_downscale2(g)
        imgs[i] = small
        x = small.astype(np.float64) / 255.0
        s += x.sum()
        s2 += (x * x).sum()
        npx += x.size
        t = soft_target(annotations, name)
        tgts[i] = target_to_uint8(area_downscale2(t))
        k = len(annotations.versions_of[name])
        cnt = np.rint(t * k).astype(np.int64)
        for c, v in zip(*np.unique(cnt, return_counts=True)):
            hist2048[(k, int(c))] = hist2048.get((k, int(c)), 0) + int(v)
        if (i + 1) % 100 == 0 or i + 1 == n:
            log("  cache: %d / %d train images, %.0f s" % (i + 1, n, time.time() - t0))
    imgs.flush()
    tgts.flush()
    del imgs, tgts

    sel_test: list[str] = []
    has_test = layout.test_images is not None
    if has_test:
        sel_test = list(test_names) if test_names is not None else list_image_names(layout.test_images)
        if test_names is None and limit is not None:
            sel_test = sel_test[:limit]
        timgs = np.lib.format.open_memmap(tmp["test"], mode="w+", dtype=np.uint8, shape=(len(sel_test), 1024, 1024))
        for i, name in enumerate(sel_test):
            g = read_gray(image_path(layout.test_images, name))
            if g.shape != (FULL_SIZE, FULL_SIZE):
                raise ValueError("%s has shape %s" % (name, g.shape))
            timgs[i] = area_downscale2(g)
        timgs.flush()
        del timgs

    arrays = ["images", "targets"] + (["test"] if has_test else [])
    sha = {}
    for key in arrays:  # fsync + rename each array, then record its checksum
        finalise_file(tmp[key], out / _CACHE_FILES[key])
        sha[_CACHE_FILES[key]] = sha256_file(out / _CACHE_FILES[key])

    mean = s / npx
    std = float(np.sqrt(s2 / npx - mean**2))
    index = {
        "cache_version": 2,
        "arrays_sha256": sha,
        "train_names": names,
        "test_names": sel_test,
        "n_versions": [len(annotations.versions_of[nm]) for nm in names],
        "instruments": [instrument_of(nm) for nm in names],
        "mean_1024": mean,
        "std_1024": std,
        "n_pixels": npx,
        "level_hist_2048": {"%d:%d" % k: v for k, v in sorted(hist2048.items())},
        "build_seconds": time.time() - t0,
    }
    atomic_write_json(out / _CACHE_FILES["index"], index)  # written last: its presence marks a complete cache
    return index


def verify_cache(cache_dir: str | Path) -> dict:
    """Recompute the SHA-256 of the cache arrays and compare with the index (cache version 2 and later).

    Returns ``{file: {"sha256": ..., "expected": ... or None, "ok": True/False/None}}``; ``ok`` is None when the index
    predates checksums (the hash is still computed, so it can be recorded).  Takes a few seconds per 0.7 GB.
    """
    d = Path(cache_dir)
    idx = json.loads((d / _CACHE_FILES["index"]).read_text(encoding="utf-8"))
    expected = idx.get("arrays_sha256", {})
    out = {}
    for key in ("images", "targets", "test"):
        f = d / _CACHE_FILES[key]
        if not f.exists():
            continue
        h = sha256_file(f)
        exp = expected.get(f.name)
        out[f.name] = {"sha256": h, "expected": exp, "ok": (h == exp) if exp else None}
    return out


def load_cache(cache_dir: str | Path) -> Cache:
    d = Path(cache_dir)
    with (d / _CACHE_FILES["index"]).open("r", encoding="utf-8") as f:
        idx = json.load(f)
    test = d / _CACHE_FILES["test"]
    return Cache(
        train_names=idx["train_names"],
        test_names=idx["test_names"],
        n_versions=idx["n_versions"],
        images_1024=np.load(d / _CACHE_FILES["images"], mmap_mode="r"),
        targets_1024=np.load(d / _CACHE_FILES["targets"], mmap_mode="r"),
        test_images_1024=np.load(test, mmap_mode="r") if test.exists() and idx["test_names"] else None,
        meta=idx,
    )


# --------------------------------------------------------------------------- augmentation switches (E6, A4.4)
@dataclass(frozen=True)
class AugmentConfig:
    """Default for E1/E1b is *none* (a reading of A6 E6, confirmed by the team: see OPEN_QUESTIONS).

    flip:        random horizontal and vertical flips (independent coins)
    rot90:       random multiple of 90 degrees
    brightness:  (lo, hi) multiplicative factor on the 0..1 intensity, clipped to [0, 1]; ``None`` = off (A4.4: 0.9..1.1)
    """

    flip: bool = False
    rot90: bool = False
    brightness: tuple[float, float] | None = None


@dataclass(frozen=True)
class SampleSpec:
    """Everything random about one training sample, drawn by the sampler (so the dataset is deterministic)."""

    index: int  # row in the dataset arrays
    side: int | None  # crop side in px at the arm's resolution; None = whole image
    y0: int = 0
    x0: int = 0
    flip_h: bool = False
    flip_v: bool = False
    rot90: int = 0
    brightness: float = 1.0


def crop_sides_for(resolution: int, crop_min: int | None, crop_max: int | None, multiple: int = 32) -> list[int] | None:
    """Allowed crop sides.  ``None`` means whole images.

    A4.2 / A4.8 item 1: at 512 px the whole image; at 1024 px a random crop of 512 to 768 px.  The U-Net's encoder has
    total stride 32, so sides are multiples of 32 (reading, see OPEN_QUESTIONS); one side is drawn per batch so the
    samples can be stacked.
    """
    if crop_min is None or crop_max is None:
        return None
    if crop_max > resolution:
        raise ValueError("crop_max %d exceeds the resolution %d" % (crop_max, resolution))
    sides = [s for s in range(crop_min, crop_max + 1) if s % multiple == 0]
    if not sides:
        raise ValueError("no multiple of %d between %d and %d" % (multiple, crop_min, crop_max))
    return sides


class SpecSampler:
    """Yields ``n_steps`` batches of :class:`SampleSpec`.

    Indices are drawn from shuffled passes over ``indices`` (no image repeats inside a pass); one crop side is drawn per
    batch; flips, rotation and brightness are drawn per sample.  The whole stream is a function of ``seed``.
    The budget is counted in steps, not epochs (A4.3).

    Note (Gate 1 review F5): a batch that straddles two passes (the leftover of one pass plus the start of the next)
    can contain the same image twice.  This is harmless for training; keep it in mind when comparing batch statistics.
    """

    def __init__(
        self,
        indices: Sequence[int],
        batch_size: int,
        n_steps: int,
        resolution: int,
        crop_sides: Sequence[int] | None,
        augment: AugmentConfig = AugmentConfig(),
        seed: int = 0,
        start_step: int = 0,
    ):
        if len(indices) == 0:
            raise ValueError("no training indices")
        if not 0 <= start_step <= n_steps:
            raise ValueError("start_step %d outside 0..%d" % (start_step, n_steps))
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.n_steps = int(n_steps)
        self.resolution = int(resolution)
        self.crop_sides = list(crop_sides) if crop_sides else None
        self.augment = augment
        self.seed = int(seed)
        self.start_step = int(start_step)  # resume: the first ``start_step`` batches are drawn and discarded

    def __len__(self) -> int:
        return self.n_steps - self.start_step

    def __iter__(self) -> Iterator[list[SampleSpec]]:
        rng = np.random.default_rng(self.seed)
        pool = np.empty(0, dtype=np.int64)
        for step in range(self.n_steps):
            while len(pool) < self.batch_size:
                pool = np.concatenate([pool, rng.permutation(self.indices)])
            take, pool = pool[: self.batch_size], pool[self.batch_size :]
            side = int(rng.choice(self.crop_sides)) if self.crop_sides else None
            batch = []
            for idx in take:
                if side is None:
                    y0 = x0 = 0
                else:
                    y0 = int(rng.integers(0, self.resolution - side + 1))
                    x0 = int(rng.integers(0, self.resolution - side + 1))
                a = self.augment
                batch.append(
                    SampleSpec(
                        index=int(idx),
                        side=side,
                        y0=y0,
                        x0=x0,
                        flip_h=bool(rng.integers(0, 2)) if a.flip else False,
                        flip_v=bool(rng.integers(0, 2)) if a.flip else False,
                        rot90=int(rng.integers(0, 4)) if a.rot90 else 0,
                        brightness=float(rng.uniform(*a.brightness)) if a.brightness else 1.0,
                    )
                )
            if step >= self.start_step:  # earlier batches were already trained before the interruption
                yield batch


class FilamentDataset(torch.utils.data.Dataset):
    """Training samples from uint8 arrays at the arm's resolution.

    ``dataset[spec]`` returns ``(image, target)`` float32 tensors of shape (1, s, s): image normalised by P1, target in
    [0, 1] (the soft mean of the annotator versions).  Use with ``DataLoader(ds, batch_sampler=SpecSampler(...))``.
    """

    def __init__(self, images_u8: np.ndarray, targets_u8: np.ndarray, mean: float, std: float):
        if images_u8.shape != targets_u8.shape:
            raise ValueError("images %s and targets %s differ" % (images_u8.shape, targets_u8.shape))
        self.images, self.targets, self.mean, self.std = images_u8, targets_u8, float(mean), float(std)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, spec: SampleSpec):
        img = np.asarray(self.images[spec.index])
        tgt = np.asarray(self.targets[spec.index])
        if spec.side is not None:
            sl = (slice(spec.y0, spec.y0 + spec.side), slice(spec.x0, spec.x0 + spec.side))
            img, tgt = img[sl], tgt[sl]
        if spec.flip_h:
            img, tgt = img[:, ::-1], tgt[:, ::-1]
        if spec.flip_v:
            img, tgt = img[::-1, :], tgt[::-1, :]
        if spec.rot90:
            img, tgt = np.rot90(img, spec.rot90), np.rot90(tgt, spec.rot90)
        x = img.astype(np.float32) / 255.0
        if spec.brightness != 1.0:
            x = np.clip(x * spec.brightness, 0.0, 1.0)  # multiplication keeps a filament's relative dip (A4.4)
        x = (x - self.mean) / self.std
        y = tgt.astype(np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(x))[None], torch.from_numpy(np.ascontiguousarray(y))[None]


class EvalDataset(torch.utils.data.Dataset):
    """Whole images at the arm's resolution (inference is always whole-image, A4.8 item 1)."""

    def __init__(self, images_u8: np.ndarray, mean: float, std: float):
        self.images, self.mean, self.std = images_u8, float(mean), float(std)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        x = normalise(np.asarray(self.images[i]), self.mean, self.std)
        return torch.from_numpy(x)[None], i
