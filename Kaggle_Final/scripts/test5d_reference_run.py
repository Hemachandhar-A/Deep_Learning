"""Test 5(d), full run: the host's own scoring functions against ``pq_metric`` on all 598 human pairs.

    python scripts/test5d_reference_run.py --data-root <folder> --out test5d_result.json [--limit N]

Adapted from the reviewer's ``test5d_reference_run.py`` (Gate 1 review, F1): the same computation, paths as arguments.
Reference functions: ``tests/host_scoring_reference.py`` (transcribed from the public Self Evaluation notebook, version
6 of 6, Apache 2.0).  Needs torch, pandas and pycocotools.  About 400 s on 2 CPU cores for all 598 pairs.

Compares, for "earlier version = GT, later version = prediction": the host functions, ``pq_metric`` with
``matching="host"`` and with ``matching="unique"``; per pair TP/FP/FN and the sum of IoU; the largest IoU difference over
all matrix cells (the host computes in float32); and the self-score of the one version with two near-identical filaments.
"""
import _bootstrap  # noqa: F401

import argparse
import collections
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as M

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import host_scoring_reference as H  # noqa: E402

from filament import data  # noqa: E402
from filament import pq_metric as P  # noqa: E402

DUPLICATE_VERSION = "010403-20160810234034Lh"


def cnt(r):
    return r["counts"].decode() if isinstance(r["counts"], bytes) else r["counts"]


def frame(ids, rles):
    return pd.DataFrame({"filament_id": ids, "segmentation_rle": [cnt(r) for r in rles]})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N pairs (dry run)")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    ann = data.load_annotations(data.find_data_layout(args.data_root).train_json)
    rles = {v: data.version_rles(a) for v, a in ann.version_anns.items()}
    pairs = [(n, v1, v2) for n in ann.image_names for v1, v2 in itertools.combinations(ann.versions_of[n], 2)]
    if args.limit:
        pairs = pairs[: args.limit]
    print("pairs", len(pairs), flush=True)

    t0 = time.time()
    overlaps, items = [], []
    max_diff, n_cells = 0.0, 0
    for k, (n, v1, v2) in enumerate(pairs):
        ov = H.get_overlap_df(frame(["%s_%d" % (v1, i + 1) for i in range(len(rles[v1]))], rles[v1]),
                              frame(["%s_%d" % (n, i + 1) for i in range(len(rles[v2]))], rles[v2]))
        overlaps.append(ov)
        it = P.ScoreItem(key=v1 + "|" + v2, image=n, gt=rles[v1], pred=rles[v2])
        items.append(it)
        iou_r, _, _ = P._iou_dice_fast(it.gt, it.pred)
        iou_h = ov["iou_matrix"].iloc[0].numpy()
        max_diff = max(max_diff, float(np.abs(iou_h.astype(np.float64) - iou_r).max()))
        n_cells += iou_h.size
        if (k + 1) % 100 == 0:
            print("  %d pairs, %.0f s" % (k + 1, time.time() - t0), flush=True)
    ov_all = pd.concat(overlaps, ignore_index=True)
    pq_host = H.get_pq_score(ov_all)
    s, tp, fp, fn = H.get_counts(ov_all)

    replica = {}
    for m in ("host", "unique"):
        r = P.score_items_fast(items, matching=m)
        replica[m] = dict(pq=r.pq, sq=r.sq, rq=r.rq, tp=r.tp, fp=r.fp, fn=r.fn, sum_iou=r.sum_iou)
    rh = P.score_items_fast(items, matching="host")
    count_mismatch = sum_mismatch = 0
    max_sum_diff = 0.0
    for ov, row in zip(overlaps, rh.rows):
        s1, t1, f1, n1 = H.get_counts(ov)
        if (t1, f1, n1) != (row.tp, row.fp, row.fn):
            count_mismatch += 1
        d = abs(s1 - row.sum_iou)
        max_sum_diff = max(max_sum_diff, d)
        sum_mismatch += d > 1e-6

    v = DUPLICATE_VERSION
    n = v.split("-", 1)[1]
    ov = H.get_overlap_df(frame(["%s_%d" % (v, i + 1) for i in range(len(rles[v]))], rles[v]),
                          frame(["%s_%d" % (n, i + 1) for i in range(len(rles[v]))], rles[v]))
    out = dict(
        n_pairs=len(pairs),
        host_function=dict(pq=pq_host, sum_iou=s, tp=tp, fp=fp, fn=fn),
        replica=replica,
        max_abs_iou_diff_float32_vs_replica=max_diff, n_iou_cells=n_cells,
        per_pair_count_mismatches_host_rule=count_mismatch,
        per_pair_sum_iou_diff_gt_1e6=int(sum_mismatch), max_per_pair_sum_iou_diff=max_sum_diff,
        self_score_010403_host_function=dict(pq=H.get_pq_score(ov), counts=H.get_counts(ov), n_filaments=len(rles[v])),
        seconds=time.time() - t0,
    )
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
