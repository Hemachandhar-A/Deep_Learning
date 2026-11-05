"""RLE encoding and the submission CSV (implements Spec S4 "Submission file", Arch section 2 step 6).

Why this module exists
----------------------
The host reads the column ``segmentation_rle`` as the pycocotools *compressed RLE counts string only*
(no quotes, no size) for a mask of fixed size 2048 x 2048, one row per predicted filament, and
``filament_id`` as ``<test image name without extension>_<k>``.  Predictions inside one image must be
pixel-disjoint, otherwise the upload ends in ERROR and burns one of the 5 daily submissions (S4, thread 735377).
So this module offers the encoder, the CSV writer, and a checker that every CSV must pass before it is
handed to the user for upload.

Conventions
-----------
* A mask is a (H, W) array of 0/1 (bool or uint8).  A stack of masks is (N, H, W).
* Encoding follows the host's decode style (Spec S4):
  ``M.encode(np.asfortranarray(mask.astype(np.uint8)))['counts'].decode('utf-8')``.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from pycocotools import mask as mask_utils

from .io_utils import atomic_open

IMAGE_SIZE = (2048, 2048)  # (H, W); fixed by the host (S4)
CSV_COLUMNS = ("filament_id", "segmentation_rle")
_ID_RE = re.compile(r"^(?P<image>.+)_(?P<k>[0-9]+)$")
# Characters that must never appear in the counts string of a CSV cell.
_BAD_CELL_CHARS = re.compile(r'[\s",]')


def _as_binary_uint8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.dtype == bool:
        return mask.astype(np.uint8)
    if mask.size and (mask.min() < 0 or mask.max() > 1):
        raise ValueError("mask must contain only 0/1 (got min %s, max %s)" % (mask.min(), mask.max()))
    return mask.astype(np.uint8)


def encode_mask(mask: np.ndarray, expected_shape: tuple[int, int] | None = IMAGE_SIZE) -> str:
    """Encode one (H, W) 0/1 mask to the pycocotools compressed-RLE counts string.

    ``expected_shape`` guards the 2048 x 2048 size the host fixes; pass ``None`` for small test masks.
    """
    mask = _as_binary_uint8(mask)
    if mask.ndim != 2:
        raise ValueError("mask must be 2-D, got shape %s" % (mask.shape,))
    if expected_shape is not None and tuple(mask.shape) != tuple(expected_shape):
        raise ValueError("mask shape %s differs from the fixed size %s" % (mask.shape, expected_shape))
    rle = mask_utils.encode(np.asfortranarray(mask))  # Fortran (column-major) order, as the host's decoder expects
    return rle["counts"].decode("utf-8")


def encode_masks(masks: np.ndarray, expected_shape: tuple[int, int] | None = IMAGE_SIZE) -> list[str]:
    """Encode a (N, H, W) stack of masks; returns N counts strings (empty list for N = 0)."""
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError("masks must be (N, H, W), got shape %s" % (masks.shape,))
    if masks.shape[0] == 0:
        return []
    masks = _as_binary_uint8(masks)
    if expected_shape is not None and tuple(masks.shape[1:]) != tuple(expected_shape):
        raise ValueError("mask shape %s differs from the fixed size %s" % (masks.shape[1:], expected_shape))
    hwn = np.asfortranarray(np.transpose(masks, (1, 2, 0)))
    return [r["counts"].decode("utf-8") for r in mask_utils.encode(hwn)]


def counts_to_rle(counts: str | bytes, size: tuple[int, int] = IMAGE_SIZE) -> dict:
    """Wrap a counts string in the dict pycocotools functions expect (size fixed at 2048 x 2048)."""
    return {"size": [int(size[0]), int(size[1])], "counts": counts.encode("utf-8") if isinstance(counts, str) else counts}


def decode_counts(counts: str | bytes, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Decode one counts string to a (H, W) uint8 mask."""
    return mask_utils.decode(counts_to_rle(counts, size))


def decode_many(counts_list: Sequence[str | bytes], size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Decode a list of counts strings to an (N, H, W) uint8 stack (the host's decode returns (H, W, N))."""
    if len(counts_list) == 0:
        return np.zeros((0, size[0], size[1]), dtype=np.uint8)
    hwn = mask_utils.decode([counts_to_rle(c, size) for c in counts_list])
    return np.ascontiguousarray(np.transpose(hwn, (2, 0, 1)))


def encode_crop_rle(crop: np.ndarray, y0: int, x0: int, shape: tuple[int, int] = IMAGE_SIZE) -> dict:
    """RLE (pycocotools dict, compressed) of a mask that is ``crop`` placed at row ``y0``, column ``x0`` of an
    all-zero ``shape`` image - without ever allocating the full image.

    The cost is proportional to the crop, not to 2048 x 2048: this is what keeps a map with thousands of small
    components cheap.  The run lengths are built in the column-major (Fortran) order the host's decoder expects, and
    the result equals ``encode_mask`` of the full mask (tested on random masks and edge cases).
    """
    H, W = int(shape[0]), int(shape[1])
    crop = np.asarray(crop)
    ys, xs = np.nonzero(crop)
    total = H * W
    if ys.size == 0:
        counts = [total]
    else:
        if y0 < 0 or x0 < 0 or y0 + crop.shape[0] > H or x0 + crop.shape[1] > W:
            raise ValueError("crop at (%d, %d) of shape %s does not fit in %s" % (y0, x0, crop.shape, shape))
        idx = np.sort((xs.astype(np.int64) + x0) * H + (ys.astype(np.int64) + y0))  # column-major linear index
        brk = np.nonzero(np.diff(idx) != 1)[0] + 1
        starts = np.concatenate(([0], brk))
        ends = np.concatenate((brk, [idx.size]))
        run_start, run_len = idx[starts], ends - starts
        prev_end = np.concatenate(([0], run_start[:-1] + run_len[:-1]))
        out = np.empty(2 * run_start.size, dtype=np.int64)
        out[0::2] = run_start - prev_end  # zeros before each run of ones
        out[1::2] = run_len
        tail = total - int(run_start[-1] + run_len[-1])
        counts = out.tolist() + ([tail] if tail > 0 else [])
    return mask_utils.frPyObjects({"size": [H, W], "counts": counts}, H, W)


def encode_crop(crop: np.ndarray, y0: int, x0: int, shape: tuple[int, int] = IMAGE_SIZE) -> str:
    """Counts string of :func:`encode_crop_rle` (what goes into the CSV)."""
    return encode_crop_rle(crop, y0, x0, shape)["counts"].decode("utf-8")


def count_overlap_pixels(masks: np.ndarray) -> int:
    """Number of pixels claimed by two or more masks of an (N, H, W) stack (0 means disjoint)."""
    masks = np.asarray(masks)
    if masks.shape[0] < 2:
        return 0
    return int((masks.astype(np.uint8).sum(axis=0) > 1).sum())


def make_filament_ids(image_name: str, n: int, start: int = 1) -> list[str]:
    """``<image>_<k>`` for k = start .. start+n-1.  ``image_name`` is the file name without extension."""
    return ["%s_%d" % (image_name, k) for k in range(start, start + n)]


def write_submission_csv(path: str | Path, rows: Iterable[tuple[str, str]]) -> int:
    """Write the CSV with columns ``filament_id,segmentation_rle``; returns the number of data rows.

    An image without predictions simply has no rows (S4); the caller just passes no row for it.
    """
    n = 0
    with atomic_open(path, "w", encoding="utf-8", newline="") as f:  # a crash never leaves a half-written submission
        w = csv.writer(f, lineterminator="\n")
        w.writerow(CSV_COLUMNS)
        for fid, counts in rows:
            w.writerow([fid, counts])
            n += 1
    return n


def read_submission_csv(path: str | Path) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (header, rows) exactly as written, with no type conversion."""
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        rows = [tuple(r) for r in reader]
    return list(header), rows


@dataclass
class CheckReport:
    """Result of :func:`check_submission_rows`; ``ok`` is True only when ``problems`` is empty."""

    n_rows: int = 0
    n_images: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        head = "CSV check: %d rows, %d images, %s" % (self.n_rows, self.n_images, "OK" if self.ok else "FAILED")
        return head if self.ok else head + "\n  " + "\n  ".join(self.problems[:50])


def check_submission_rows(
    header: Sequence[str],
    rows: Sequence[tuple[str, str]],
    expected_images: Iterable[str] | None = None,
    size: tuple[int, int] = IMAGE_SIZE,
) -> CheckReport:
    """All checks of Spec S4 that cost no submission slot.

    1. header is exactly ``filament_id,segmentation_rle``;
    2. every id is ``<image>_<k>`` with integer k, and ``<image>`` is a known test image (if given);
    3. ids are unique;
    4. the counts string is non-empty and has no whitespace, quote or comma;
    5. it decodes at the fixed size, has at least one pixel, and re-encodes to the same string;
    6. inside one image no pixel is claimed by two predictions (the check that protects the daily slot).
    """
    rep = CheckReport(n_rows=len(rows))
    if tuple(header) != CSV_COLUMNS:
        rep.problems.append("header is %r, expected %r" % (tuple(header), CSV_COLUMNS))
    expected = set(expected_images) if expected_images is not None else None

    seen: set[str] = set()
    by_image: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for i, row in enumerate(rows):
        if len(row) != 2:
            rep.problems.append("row %d has %d cells, expected 2" % (i, len(row)))
            continue
        fid, counts = row
        m = _ID_RE.match(fid)
        if not m:
            rep.problems.append("row %d: id %r does not look like <image>_<k>" % (i, fid))
            continue
        image = m.group("image")
        if expected is not None and image not in expected:
            rep.problems.append("row %d: image %r is not a known test image" % (i, image))
        if fid in seen:
            rep.problems.append("row %d: duplicate id %r" % (i, fid))
        seen.add(fid)
        if not counts or _BAD_CELL_CHARS.search(counts):
            rep.problems.append("row %d (%s): counts string empty or contains whitespace/quote/comma" % (i, fid))
            continue
        by_image[image].append((fid, counts))
    rep.n_images = len(by_image)

    for image, items in by_image.items():
        ok_rles: list[dict] = []
        ok_ids: list[str] = []
        for fid, counts in items:
            try:
                m = decode_counts(counts, size)  # one mask at a time: nothing is kept in memory
            except Exception as exc:  # pycocotools raises plain exceptions on malformed counts
                rep.problems.append("%s: cannot decode (%s)" % (fid, type(exc).__name__))
                continue
            if m.sum() == 0:
                rep.problems.append("%s: empty mask" % fid)
            if encode_mask(m, expected_shape=size) != counts:
                rep.problems.append("%s: re-encoding the decoded mask gives a different string" % fid)
            ok_rles.append(counts_to_rle(counts, size))
            ok_ids.append(fid)
        # Overlap, exactly and without decoding all masks together: the union of N masks has the area of the sum of
        # their areas if and only if no pixel is claimed twice.  Only when that fails are the masks stacked, to count.
        if len(ok_rles) > 1:
            total = sum(int(mask_utils.area(r)) for r in ok_rles)
            union = int(mask_utils.area(mask_utils.merge(ok_rles)))
            if union != total:
                n_over = count_overlap_pixels(decode_many([r["counts"] for r in ok_rles], size))
                rep.problems.append("%s: %d pixels are claimed by two or more predictions" % (image, n_over))
    return rep


def check_submission_csv(
    path: str | Path, expected_images: Iterable[str] | None = None, size: tuple[int, int] = IMAGE_SIZE
) -> CheckReport:
    """Read a CSV from disk and run :func:`check_submission_rows`."""
    header, rows = read_submission_csv(path)
    return check_submission_rows(header, rows, expected_images, size)


def verify_round_trip(path: str | Path, masks_by_id: Mapping[str, np.ndarray]) -> list[str]:
    """Compare the CSV's decoded masks with the masks that were meant to be written.

    Returns a list of problems (empty = every id present, no extra id, every mask pixel-identical).
    """
    _, rows = read_submission_csv(path)
    problems: list[str] = []
    got = {fid: counts for fid, counts in rows}
    for fid in masks_by_id:
        if fid not in got:
            problems.append("%s missing from CSV" % fid)
    for fid, counts in got.items():
        if fid not in masks_by_id:
            problems.append("%s in CSV but not expected" % fid)
        elif not np.array_equal(decode_counts(counts), _as_binary_uint8(masks_by_id[fid])):
            problems.append("%s decodes to a different mask" % fid)
    return problems
