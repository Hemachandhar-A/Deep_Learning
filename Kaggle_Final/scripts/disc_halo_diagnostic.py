"""Diagnostic for the disc recipe (acceptance test 8): how often does the fit follow a scattered-light halo?

    python scripts/disc_halo_diagnostic.py --data-root <folder>

Prints the radius distribution of the documented recipe (threshold fraction 0.65) and how it changes with the fraction.
It does NOT change the recipe: the evidence is for the reviewer (OPEN_QUESTIONS, disc).
"""
import _bootstrap  # noqa: F401

import argparse
import collections
from pathlib import Path

import cv2
import numpy as np

from filament import data

FRACTIONS = [0.65, 0.70, 0.75, 0.80, 0.85]


def _largest(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    args = ap.parse_args()
    layout = data.find_data_layout(args.data_root)
    names = data.list_image_names(layout.train_images)
    rad = {f: [] for f in FRACTIONS}
    for nm in names:
        g = data.read_gray(data.image_path(layout.train_images, nm))
        small = cv2.resize(g, (512, 512), interpolation=cv2.INTER_AREA)
        _, bw = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        med = float(np.median(small[_largest(bw)]))
        for f in FRACTIONS:
            rad[f].append(4 * np.sqrt(_largest(small > f * med).sum() / np.pi))
    r = {f: np.array(v) for f, v in rad.items()}
    arr = np.array(names)
    print("radius at fraction 0.65 (documented recipe) over %d images: quantiles 5/50/95/99/max %s"
          % (len(names), np.round(np.percentile(r[0.65], [5, 50, 95, 99, 100]), 1)))
    for thr in (910, 920, 940, 960):
        sel = r[0.65] > thr
        c = collections.Counter(data.instrument_of(n) for n in arr[sel])
        print("  radius > %d px: %3d images %s" % (thr, sel.sum(), dict(sorted(c.items()))))
    halo = r[0.65] > 940
    print("fraction | median   p5    p95   max | images > 920 px | median over the %d images that are > 940 px at 0.65" % halo.sum())
    for f in FRACTIONS:
        print("  %.2f   | %6.1f %6.1f %6.1f %6.1f | %4d | %6.1f" % (
            f, np.median(r[f]), *np.percentile(r[f], [5, 95]), r[f].max(), (r[f] > 920).sum(), np.median(r[f][halo])))


if __name__ == "__main__":
    main()
