"""Build the 1024 px cache (Arch P1, P2, P7, P8; A4.8 items 2 and 3).

    python scripts/prepare_cache.py --data-root <folder with the competition files> --cache-dir <empty output folder>

Reads the JPEGs and the annotation JSON (never writes there), rasterises every filament with pycocotools, builds the
soft target (mean of the annotator versions), and saves uint8 images and targets at 1024 px as .npy files.  Prints the
build time (A4.8 item 2), the recomputed normalisation constants (P1) and the histogram of soft-target levels.
"""
import _bootstrap  # noqa: F401  (adds src/ to sys.path)

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from filament import data

DOC_MEAN, DOC_STD = 0.32742, 0.22230  # Spec S5 (norm_stats.py)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="build only the first N training images (dry run)")
    ap.add_argument("--select-train", type=int, default=None,
                    help="dry run: N training images spread over the instruments (seeded), instead of all 707")
    ap.add_argument("--select-test", type=int, default=None, help="dry run: M test images (seeded), instead of all 180")
    ap.add_argument("--select-seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true",
                    help="only recompute the SHA-256 of an existing cache and compare with its index (risk K15)")
    args = ap.parse_args()
    if args.verify:
        res = data.verify_cache(args.cache_dir)
        for f, r in res.items():
            print("%-26s sha256 %s  index: %s" % (f, r["sha256"], {True: "MATCH", False: "DIFFERENT", None: "no checksum in index"}[r["ok"]]))
        raise SystemExit(0 if all(r["ok"] is not False for r in res.values()) else 1)

    layout = data.find_data_layout(args.data_root)
    print("train json  :", layout.train_json)
    print("train images:", layout.train_images)
    print("test images :", layout.test_images)
    t0 = time.time()
    ann = data.load_annotations(layout.train_json)
    print("annotations : %d images, %d versions, %d filaments (loaded in %.1f s)"
          % (len(ann.image_names), ann.n_versions, ann.n_filaments, time.time() - t0))

    sel_train = sel_test = None
    if args.select_train:
        sel_train = data.select_subset(ann.image_names, args.select_train, args.select_seed)
        print("dry run: %d training images selected (seed %d)" % (len(sel_train), args.select_seed))
    if args.select_test and layout.test_images is not None:
        sel_test = data.select_subset(data.list_image_names(layout.test_images), args.select_test, args.select_seed)
        print("dry run: %d test images selected" % len(sel_test))
    idx = data.build_cache(layout, ann, args.cache_dir, limit=args.limit, names=sel_train, test_names=sel_test)
    print("cache build time: %.1f s" % idx["build_seconds"])
    print("train images %d, test images %d" % (len(idx["train_names"]), len(idx["test_names"])))
    print("normalisation at 1024 px over %d pixels: mean %.5f  std %.5f   (document: %.5f, %.5f)"
          % (idx["n_pixels"], idx["mean_1024"], idx["std_1024"], DOC_MEAN, DOC_STD))

    levels = defaultdict(int)
    for key, n in idx["level_hist_2048"].items():
        k, c = (int(x) for x in key.split(":"))
        levels[round(c / k, 6)] += n
    total = sum(levels.values())
    print("soft-target levels at 2048 px (value: pixels, share):")
    for v in sorted(levels):
        print("  %.4f: %12d  %.5f" % (v, levels[v], levels[v] / total))

    out = args.cache_dir
    for f in sorted(out.glob("*.npy")):
        print("  %-28s %7.1f MB  sha256 %s" % (f.name, f.stat().st_size / 1e6, idx["arrays_sha256"].get(f.name, "")[:16]))


if __name__ == "__main__":
    main()
