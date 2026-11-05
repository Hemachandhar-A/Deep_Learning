"""Replica of the host's Panoptic Quality scorer (Spec S4; Arch section 4.5, risk K1).

The leaderboard metric is PQ as implemented in the host's "Self Evaluation" notebook.  Its logic is followed literally
here.  It was read from the rendered notebook on 2026-10-11 (S4) and, in the Gate 1 review (2026-10-12), from the public
notebook page (version 6 of 6, Apache 2.0): ``get_pq_score`` sets ``hit_matrix = iou_matrix > 0.5``, adds every IoU at a
hit cell to the true positives, counts a prediction column without a hit as FP and a GT row without a hit as FN.  It was
**confirmed numerically**: the host's own functions (tests/host_scoring_reference.py) and ``matching="host"`` give the
same TP, FP, FN and PQ on all 598 human pairs, pair by pair (tests/test_host_reference.py,
scripts/test5d_reference_run.py).  The rule is:

1. Ground truth has one row per filament; rows are grouped into *annotator-images* (one annotator version of one image).
   Predictions are grouped by image.
2. For every annotator-image: stack its n_gt GT masks and all n_pred predicted masks of the same image and build the
   n_gt x n_pred IoU matrix from flattened masks (a union of zero gives IoU 0); a second matrix holds pairwise Dice.
3. A hit is IoU > 0.5 (strict).  TP IoUs are the IoUs at hits; FP = prediction columns without a hit; FN = GT rows
   without a hit.  If n_pred = 0 all GT rows are FN.  Images that appear only in the predictions are ignored.
4. PQ = sum(TP IoUs) / (TP + 0.5 FP + 0.5 FN), pooled over all annotator-images (micro average); 0.0 when the
   denominator is 0.
5. Secondary distributions: Dice over pairs with Dice > 0, IoU, and one-to-many / many-to-one counts over pairs with
   IoU > 0.

Two implementations share the accumulation code and differ only in how the IoU matrix is produced:

* ``score_reference``  dense masks, intersection by matrix product of flattened masks (exact integer arithmetic);
* ``score_fast``       RLEs and ``pycocotools.mask.iou``; used when many settings are tuned.

Tests (tests/test_pq_metric.py) check that both give identical TP/FP/FN and PQ on random cases before the fast one is
used anywhere.

Matching rule (``matching=``)
-----------------------------
* ``"host"`` (default): every cell with IoU > 0.5 is a hit, exactly as the notebook's code (confirmed, see above).  With pixel-disjoint
  predictions each GT filament has at most one hit (Kirillov et al., Theorem 1) *unless two GT filaments of one version
  overlap*.  This is the competition metric.
* ``"unique"``: greedy one-to-one matching by descending IoU, the rule of the analysis scripts (stats2_chunk.py) that
  produced the human-agreement numbers of S5 (TP 2,144).  Differs from "host" only when one prediction/GT filament has two
  hits.  Measured on the training data: the single version 010403-20160810234034Lh contains two filaments with
  IoU 0.724 to each other; scoring it against itself gives PQ 0.9448 under "host" and 1.0 under "unique", and the
  598 human pairs give TP 2,146 / FP 1,849 under "host" against 2,144 / 1,851 under "unique" (PQ, SQ, RQ equal to 3
  decimals).  See tests/test_pq_metric.py and tests/test_host_reference.py.

A GT version with zero filaments is skipped here.  The host function has an ``n_gt == 0`` branch that would count all
predictions as FP, but annotator-images are taken from the ids of the GT table, so the branch cannot be reached; the
competition data have no empty version either (S3).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from pycocotools import mask as mask_utils

HIT_IOU = 0.5  # strict: a hit is IoU > 0.5


@dataclass
class ScoreItem:
    """One annotator-image to score.  ``gt`` and ``pred`` are lists of RLE dicts (fast) or an (N, H, W) array (reference)."""

    key: str  # annotator-image (version id) or any unique key
    image: str  # image name, the unit of the bootstrap
    gt: object
    pred: object


@dataclass
class VersionRow:
    key: str
    image: str
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    sum_iou: float
    pixel_dice: float  # Dice of the union of GT masks against the union of predicted masks


@dataclass
class PQResult:
    pq: float
    sq: float
    rq: float
    tp: int
    fp: int
    fn: int
    sum_iou: float
    rows: list[VersionRow]
    iou_matched: np.ndarray  # IoU at the hits
    iou_pairs: np.ndarray  # IoU of all (gt, pred) pairs with IoU > 0
    dice_pairs: np.ndarray  # Dice of all (gt, pred) pairs with Dice > 0
    gt_partners: np.ndarray  # per GT filament: number of predictions with IoU > 0 (one-to-many)
    pred_partners: np.ndarray  # per prediction: number of GT filaments with IoU > 0 (many-to-one)

    @property
    def n_versions(self) -> int:
        return len(self.rows)

    @property
    def n_images(self) -> int:
        return len({r.image for r in self.rows})

    @property
    def mean_pixel_dice(self) -> float:
        return float(np.mean([r.pixel_dice for r in self.rows])) if self.rows else 0.0

    @property
    def pred_per_image(self) -> dict[str, int]:
        d: dict[str, int] = {}
        for r in self.rows:
            d[r.image] = r.n_pred
        return d

    def one_to_many_hist(self) -> dict[int, int]:
        """Histogram: number of predictions touching one GT filament -> count of GT filaments."""
        return dict(sorted(Counter(self.gt_partners.tolist()).items()))

    def many_to_one_hist(self) -> dict[int, int]:
        """Histogram: number of GT filaments touching one prediction -> count of predictions."""
        return dict(sorted(Counter(self.pred_partners.tolist()).items()))

    def per_image(self) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sums per image (all of its versions together): names, sum_iou, tp, fp, fn.  Input of the bootstrap."""
        agg: dict[str, list[float]] = {}
        for r in self.rows:
            a = agg.setdefault(r.image, [0.0, 0, 0, 0])
            a[0] += r.sum_iou
            a[1] += r.tp
            a[2] += r.fp
            a[3] += r.fn
        names = sorted(agg)
        arr = np.array([agg[n] for n in names], dtype=np.float64).reshape(len(names), 4)
        return names, arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    def summary(self) -> dict:
        return {
            "PQ": self.pq, "SQ": self.sq, "RQ": self.rq, "TP": self.tp, "FP": self.fp, "FN": self.fn,
            "n_versions": self.n_versions, "n_images": self.n_images,
            "mean_dice_pairs": float(self.dice_pairs.mean()) if len(self.dice_pairs) else 0.0,
            "mean_iou_matched": float(self.iou_matched.mean()) if len(self.iou_matched) else 0.0,
            "pixel_dice_mean": self.mean_pixel_dice,
            "pred_per_image_mean": float(np.mean(list(self.pred_per_image.values()))) if self.rows else 0.0,
            "one_to_many_gt_with_gt1": int((self.gt_partners > 1).sum()),
            "many_to_one_pred_with_gt1": int((self.pred_partners > 1).sum()),
        }


def pq_from_sums(sum_iou: float, tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    """(PQ, SQ, RQ) from pooled sums; 0.0 when the denominator is 0 (S4 item 4)."""
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = sum_iou / denom if denom > 0 else 0.0
    sq = sum_iou / tp if tp > 0 else 0.0
    rq = tp / denom if denom > 0 else 0.0
    return pq, sq, rq


# --------------------------------------------------------------------------- IoU providers
def _iou_dice_reference(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """n_gt x n_pred IoU and Dice matrices from flattened masks; and the pixel Dice of the two unions."""
    n_gt, n_pred = gt.shape[0], pred.shape[0]
    g = gt.reshape(n_gt, -1).astype(np.float32)
    p = pred.reshape(n_pred, -1).astype(np.float32) if n_pred else np.zeros((0, g.shape[1]), np.float32)
    # float32 sums of 0/1 are exact below 2**24 = 16.7 M; a 2048 x 2048 mask has 4.19 M pixels
    inter = (g @ p.T).astype(np.float64) if n_pred else np.zeros((n_gt, 0))
    a_g = g.sum(1).astype(np.float64)
    a_p = p.sum(1).astype(np.float64)
    union = a_g[:, None] + a_p[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
        tot = a_g[:, None] + a_p[None, :]
        dice = np.where(tot > 0, 2.0 * inter / tot, 0.0)
    ug = g.max(0) > 0
    up = (p.max(0) > 0) if n_pred else np.zeros_like(ug)
    s = ug.sum() + up.sum()
    pix = float(2.0 * (ug & up).sum() / s) if s > 0 else 0.0
    return iou, dice, pix


def _iou_dice_fast(gt: Sequence[dict], pred: Sequence[dict]) -> tuple[np.ndarray, np.ndarray, float]:
    n_gt, n_pred = len(gt), len(pred)
    if n_pred == 0:
        return np.zeros((n_gt, 0)), np.zeros((n_gt, 0)), 0.0
    iou = np.asarray(mask_utils.iou(list(gt), list(pred), [0] * n_pred), dtype=np.float64).reshape(n_gt, n_pred)
    dice = 2.0 * iou / (1.0 + iou)  # Dice = 2 IoU / (1 + IoU) for one pair; zero exactly when IoU is zero
    ug, up = mask_utils.merge(list(gt)), mask_utils.merge(list(pred))
    a, b = float(mask_utils.area(ug)), float(mask_utils.area(up))
    inter = float(mask_utils.area(mask_utils.merge([ug, up], intersect=True)))
    return iou, dice, (2.0 * inter / (a + b) if a + b > 0 else 0.0)


# --------------------------------------------------------------------------- accumulation (host rules)
def _unique_hits(iou: np.ndarray) -> np.ndarray:
    """Boolean hit matrix of the greedy one-to-one rule: highest IoU first, each row and column used once."""
    hit = np.zeros(iou.shape, dtype=bool)
    rr, cc = np.nonzero(iou > HIT_IOU)
    order = np.lexsort((cc, rr, -iou[rr, cc]))  # by -IoU, then row, then column
    used_r, used_c = set(), set()
    for k in order:
        r, c = int(rr[k]), int(cc[k])
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        hit[r, c] = True
    return hit


def _accumulate(per_item: list[tuple[ScoreItem, int, int, np.ndarray, np.ndarray, float]],
                matching: str = "host") -> PQResult:
    if matching not in ("host", "unique"):
        raise ValueError("matching must be 'host' or 'unique', got %r" % (matching,))
    rows: list[VersionRow] = []
    tp_ious: list[np.ndarray] = []
    iou_pairs: list[np.ndarray] = []
    dice_pairs: list[np.ndarray] = []
    gtp: list[np.ndarray] = []
    prp: list[np.ndarray] = []
    TP = FP = FN = 0
    SI = 0.0
    for item, n_gt, n_pred, iou, dice, pix in per_item:
        if n_gt == 0:
            continue  # not in the host's table (see module docstring)
        if n_pred == 0:
            tp, fp, fn, si = 0, 0, n_gt, 0.0
            gtp.append(np.zeros(n_gt, dtype=np.int64))
        else:
            hit = (iou > HIT_IOU) if matching == "host" else _unique_hits(iou)
            tp_ious.append(iou[hit])
            tp = int(hit.sum())
            fp = int((hit.sum(axis=0) == 0).sum())
            fn = int((hit.sum(axis=1) == 0).sum())
            si = float(iou[hit].sum())
            iou_pairs.append(iou[iou > 0])
            dice_pairs.append(dice[dice > 0])
            gtp.append((iou > 0).sum(axis=1).astype(np.int64))
            prp.append((iou > 0).sum(axis=0).astype(np.int64))
        rows.append(VersionRow(item.key, item.image, n_gt, n_pred, tp, fp, fn, si, pix))
        TP, FP, FN, SI = TP + tp, FP + fp, FN + fn, SI + si
    pq, sq, rq = pq_from_sums(SI, TP, FP, FN)
    cat = lambda xs, dt=np.float64: np.concatenate(xs) if xs else np.zeros(0, dtype=dt)  # noqa: E731
    return PQResult(
        pq=pq, sq=sq, rq=rq, tp=TP, fp=FP, fn=FN, sum_iou=SI, rows=rows,
        iou_matched=cat(tp_ious), iou_pairs=cat(iou_pairs), dice_pairs=cat(dice_pairs),
        gt_partners=cat(gtp, np.int64), pred_partners=cat(prp, np.int64),
    )


def score_items_reference(items: Sequence[ScoreItem], matching: str = "host") -> PQResult:
    """Reference implementation.  ``item.gt`` and ``item.pred`` are (N, H, W) 0/1 arrays (N may be 0 for pred)."""
    out = []
    for it in items:
        gt = np.asarray(it.gt)
        pred = np.asarray(it.pred)
        if gt.shape[0] == 0:  # no row in the host's table; skipped by _accumulate
            out.append((it, 0, int(pred.shape[0]), np.zeros((0, 0)), np.zeros((0, 0)), 0.0))
            continue
        if pred.size == 0:
            pred = np.zeros((0,) + gt.shape[1:], dtype=np.uint8)
        iou, dice, pix = _iou_dice_reference(gt, pred)
        out.append((it, gt.shape[0], pred.shape[0], iou, dice, pix))
    return _accumulate(out, matching)


def score_items_fast(items: Sequence[ScoreItem], matching: str = "host") -> PQResult:
    """Fast implementation.  ``item.gt`` and ``item.pred`` are lists of RLE dicts."""
    out = []
    for it in items:
        if len(it.gt) == 0:  # no row in the host's table; skipped by _accumulate
            out.append((it, 0, len(it.pred), np.zeros((0, 0)), np.zeros((0, 0)), 0.0))
            continue
        iou, dice, pix = _iou_dice_fast(it.gt, it.pred)
        out.append((it, len(it.gt), len(it.pred), iou, dice, pix))
    return _accumulate(out, matching)


def _items_from_dicts(gt: Mapping[str, object], preds: Mapping[str, object], version_image: Mapping[str, str], empty):
    items = []
    for key in gt:
        image = version_image[key]
        items.append(ScoreItem(key=key, image=image, gt=gt[key], pred=preds.get(image, empty)))
    return items  # prediction images without any GT version are ignored (S4 item 3)


def score_reference(gt_masks: Mapping[str, np.ndarray], pred_masks: Mapping[str, np.ndarray],
                    version_image: Mapping[str, str], matching: str = "host") -> PQResult:
    """``gt_masks[version_id]`` (n_gt, H, W); ``pred_masks[image]`` (n_pred, H, W); a missing image has no prediction."""
    return score_items_reference(
        _items_from_dicts(gt_masks, pred_masks, version_image, np.zeros((0, 1, 1), np.uint8)), matching)


def score_fast(gt_rles: Mapping[str, Sequence[dict]], pred_rles: Mapping[str, Sequence[dict]],
               version_image: Mapping[str, str], matching: str = "host") -> PQResult:
    """``gt_rles[version_id]`` list of RLEs; ``pred_rles[image]`` list of RLEs; a missing image has no prediction."""
    return score_items_fast(_items_from_dicts(gt_rles, pred_rles, version_image, []), matching)


def masks_to_rles(masks: np.ndarray) -> list[dict]:
    """(N, H, W) 0/1 -> list of N pycocotools RLEs (helper for tests and for score_oof)."""
    masks = np.asarray(masks)
    if masks.shape[0] == 0:
        return []
    return mask_utils.encode(np.asfortranarray(np.transpose(masks.astype(np.uint8), (1, 2, 0))))


# --------------------------------------------------------------------------- bootstrap (A4.5, A6)
def _resample_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _pq_of_resamples(si, tp, fp, fn, idx) -> np.ndarray:
    num = si[idx].sum(axis=1)
    den = tp[idx].sum(axis=1) + 0.5 * fp[idx].sum(axis=1) + 0.5 * fn[idx].sum(axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def bootstrap_pq(res: PQResult, n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> dict:
    """Image-level bootstrap of the pooled PQ: resample *images* (all versions of an image stay together)."""
    names, si, tp, fp, fn = res.per_image()
    idx = _resample_indices(len(names), n_boot, seed)
    b = _pq_of_resamples(si, tp, fp, fn, idx)
    lo, hi = np.percentile(b, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return {"pq": res.pq, "lo": float(lo), "hi": float(hi), "n_boot": n_boot, "n_images": len(names)}


def paired_bootstrap_diff(res_a: PQResult, res_b: PQResult, n_boot: int = 1000, seed: int = 0,
                          level: float = 0.95) -> dict:
    """PQ(a) - PQ(b) with an image-level paired bootstrap: the same resampled images are used for both runs."""
    na, sa, ta, fpa, fna = res_a.per_image()
    nb, sb, tb, fpb, fnb = res_b.per_image()
    if na != nb:
        raise ValueError("the two runs cover different images")
    idx = _resample_indices(len(na), n_boot, seed)
    d = _pq_of_resamples(sa, ta, fpa, fna, idx) - _pq_of_resamples(sb, tb, fpb, fnb, idx)
    lo, hi = np.percentile(d, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return {"diff": res_a.pq - res_b.pq, "lo": float(lo), "hi": float(hi), "n_boot": n_boot,
            "p_a_gt_b": float((d > 0).mean())}
