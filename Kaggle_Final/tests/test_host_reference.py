"""Acceptance test 5(d): the host's own scoring functions against the replica (Gate 1 review, finding F1).

``tests/host_scoring_reference.py`` is the host's scoring code, transcribed by the reviewer from the public Self
Evaluation notebook (version 6 of 6, Apache 2.0, see that file's header); it is used here unchanged as the reference.
The full 598-pair comparison takes about 400 s and is run by ``scripts/test5d_reference_run.py``; the suite uses the
first 60 pairs and a synthetic case so that it stays fast.

Reviewer's 598-pair result (earlier version = GT): host functions PQ 0.339916, TP 2,146, FP 1,849, FN 1,860, sum of IoU
1359.83321 = ``pq_metric`` with ``matching="host"``; 0 of 598 pairs differ in TP, FP or FN; largest IoU difference over
35,120 cells 3.0e-8 (the host computes in float32).  That is why ``"host"`` is the default of ``pq_metric``.
"""
import itertools

import numpy as np
import pytest

host = pytest.importorskip("host_scoring_reference", reason="needs pandas and torch")
import pandas as pd  # noqa: E402

from filament import pq_metric as pq  # noqa: E402
from filament import rle  # noqa: E402

DUPLICATE_VERSION = "010403-20160810234034Lh"


def _frame(ids, rles):
    return pd.DataFrame({"filament_id": ids, "segmentation_rle": rles})


def _counts(r):
    return r["counts"].decode() if isinstance(r["counts"], bytes) else r["counts"]


def _compare(pairs, gt_rles):
    """pairs: [(image, v_gt, v_pred)].  Returns (host totals, replica result, per-pair mismatch count, max IoU diff)."""
    overlaps, items = [], []
    max_iou_diff = 0.0
    for image, v1, v2 in pairs:
        gt_df = _frame(["%s_%d" % (v1, i + 1) for i in range(len(gt_rles[v1]))], [_counts(r) for r in gt_rles[v1]])
        pr_df = _frame(["%s_%d" % (image, i + 1) for i in range(len(gt_rles[v2]))], [_counts(r) for r in gt_rles[v2]])
        ov = host.get_overlap_df(gt_df, pr_df)
        overlaps.append(ov)
        it = pq.ScoreItem("%s|%s" % (v1, v2), image, gt_rles[v1], gt_rles[v2])
        items.append(it)
        iou_r, _, _ = pq._iou_dice_fast(it.gt, it.pred)
        iou_h = ov["iou_matrix"].iloc[0].numpy().astype(np.float64)
        max_iou_diff = max(max_iou_diff, float(np.abs(iou_h - iou_r).max()))
    ov_all = pd.concat(overlaps, ignore_index=True)
    totals = {"pq": host.get_pq_score(ov_all)}
    totals["sum_iou"], totals["tp"], totals["fp"], totals["fn"] = host.get_counts(ov_all)
    res = pq.score_items_fast(items, matching="host")
    mismatch = 0
    for ov, row in zip(overlaps, res.rows):
        s, tp, fp, fn = host.get_counts(ov)
        if (tp, fp, fn) != (row.tp, row.fp, row.fn) or abs(s - row.sum_iou) > 1e-6:
            mismatch += 1
    return totals, res, mismatch, max_iou_diff


def test_host_functions_equal_replica_on_synthetic_masks():
    """No competition data needed: random rectangles at 2048 px through both implementations."""
    rng = np.random.default_rng(7)
    H = W = 2048

    def rects(n):
        out = []
        for _ in range(n):
            m = np.zeros((H, W), np.uint8)
            y, x = rng.integers(0, 1800, 2)
            h, w = rng.integers(20, 200, 2)
            m[y : y + h, x : x + w] = 1
            out.append(m)
        return out

    gt_rles, pairs = {}, []
    for k in range(4):
        base = rects(int(rng.integers(2, 6)))
        gt = base
        pred = [np.roll(m, int(rng.integers(0, 12)), axis=1) for m in base] + rects(int(rng.integers(0, 3)))
        gt_rles["g%d-img%d" % (k, k)] = [pq.masks_to_rles(m[None])[0] for m in gt]
        gt_rles["p%d-img%d" % (k, k)] = [pq.masks_to_rles(m[None])[0] for m in pred]
        pairs.append(("img%d" % k, "g%d-img%d" % (k, k), "p%d-img%d" % (k, k)))
    totals, res, mismatch, max_iou_diff = _compare(pairs, gt_rles)
    assert mismatch == 0 and max_iou_diff < 1e-6
    assert (res.tp, res.fp, res.fn) == (totals["tp"], totals["fp"], totals["fn"])
    assert res.pq == pytest.approx(totals["pq"], abs=1e-6)
    assert res.tp > 0 and res.fp + res.fn > 0  # the case exercised hits, FPs and FNs


@pytest.mark.data
def test_5d_host_functions_equal_replica_on_60_human_pairs(annotations, gt_rles):
    pairs = [(n, a, b) for n in annotations.image_names for a, b in itertools.combinations(annotations.versions_of[n], 2)]
    sub = pairs[:60]
    totals, res, mismatch, max_iou_diff = _compare(sub, gt_rles)
    print("\n[E0-5d] first %d of %d human pairs, host functions vs pq_metric(matching='host'):" % (len(sub), len(pairs)))
    print("  host    : PQ %.6f  TP %d FP %d FN %d  sum IoU %.5f" % (totals["pq"], totals["tp"], totals["fp"], totals["fn"],
                                                                 totals["sum_iou"]))
    print("  replica : PQ %.6f  TP %d FP %d FN %d  sum IoU %.5f" % (res.pq, res.tp, res.fp, res.fn, res.sum_iou))
    print("  pairs differing in TP/FP/FN/sum IoU: %d; largest IoU difference over all cells: %.2e (float32 vs float64)"
          % (mismatch, max_iou_diff))
    assert len(sub) == 60
    assert mismatch == 0 and max_iou_diff < 1e-6
    assert (res.tp, res.fp, res.fn) == (totals["tp"], totals["fp"], totals["fn"])
    assert res.pq == pytest.approx(totals["pq"], abs=1e-6)
    assert res.sum_iou == pytest.approx(totals["sum_iou"], abs=1e-4)


@pytest.mark.data
def test_5d_duplicate_version_scores_below_one_under_the_host_function(annotations, gt_rles):
    """The host's function scores version 010403-20160810234034Lh against itself at PQ 0.944835 (TP 10, 8 filaments)."""
    v = DUPLICATE_VERSION
    image = annotations.image_of[v]
    gt_df = _frame(["%s_%d" % (v, i + 1) for i in range(len(gt_rles[v]))], [_counts(r) for r in gt_rles[v]])
    pr_df = _frame(["%s_%d" % (image, i + 1) for i in range(len(gt_rles[v]))], [_counts(r) for r in gt_rles[v]])
    ov = host.get_overlap_df(gt_df, pr_df)
    s, tp, fp, fn = host.get_counts(ov)
    print("\n[E0-5d] %s against itself, host function: PQ %.6f, TP %d FP %d FN %d, %d filaments"
          % (v, host.get_pq_score(ov), tp, fp, fn, len(gt_rles[v])))
    assert (tp, fp, fn, len(gt_rles[v])) == (10, 0, 0, 8)
    assert host.get_pq_score(ov) == pytest.approx(0.944835, abs=1e-6)
    item = pq.ScoreItem(v, image, gt_rles[v], gt_rles[v])
    assert pq.score_items_fast([item], matching="host").pq == pytest.approx(host.get_pq_score(ov), abs=1e-6)
    assert pq.score_items_fast([item], matching="unique").pq == 1.0


def test_encoding_used_for_the_reference_is_the_submission_encoding():
    """The host decodes counts strings of size 2048 x 2048; the strings built here must be exactly those of rle.py."""
    m = np.zeros((2048, 2048), np.uint8)
    m[100:150, 300:420] = 1
    r = pq.masks_to_rles(m[None])[0]
    assert _counts(r) == rle.encode_mask(m)
