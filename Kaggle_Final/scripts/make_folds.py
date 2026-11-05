"""Write folds.csv once (Arch section 4.1; decided by the user on 2026-10-12, Gate 1 review D-a).

    python scripts/make_folds.py --data-root <folder> --out <work folder>/folds.csv

Default: strategy ``balanced`` (one row per image, label = instrument x number of versions, StratifiedKFold, shuffle on),
seed 42, 5 folds.  The same file is then reused by every experiment (A4.8 item 5).  An existing file is never overwritten.

folds.csv lists competition file names, so it is *not* committed (D-b).  This script prints its SHA-256 and the
scikit-learn version and writes them next to it as ``<name>.provenance.json``; every run records both in its
``metrics.json`` and checks the hash against the config before using the file.
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import sklearn

from filament import data, folds
from filament.io_utils import atomic_write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=folds.DEFAULT_SEED, help="fold seed (decided: 42)")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--strategy", choices=["balanced", "document"], default=folds.DEFAULT_STRATEGY,
                    help="'balanced' (decided, default) or 'document' (version rows, StratifiedGroupKFold; comparison only)")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit("%s exists; folds.csv is written once and reused (delete it by hand if you really mean to)" % args.out)

    layout = data.find_data_layout(args.data_root)
    ann = data.load_annotations(layout.train_json)
    rows = folds.make_folds(ann.versions_of, args.n_splits, args.seed, args.strategy)
    problems = folds.check_folds(rows, ann.versions_of, args.n_splits)
    if problems:
        raise SystemExit("fold check failed: %s" % problems)
    folds.write_folds_csv(args.out, rows)

    prov = folds.provenance(args.out)
    prov.update({"strategy": args.strategy, "seed": args.seed, "n_splits": args.n_splits, "n_images": len(rows),
                 **folds.spreads(rows)})
    sidecar = args.out.with_name(args.out.name + ".provenance.json")
    atomic_write_json(sidecar, prov)

    print("wrote %s (%d images; strategy %s; seed %d; scikit-learn %s)"
          % (args.out, len(rows), args.strategy, args.seed, sklearn.__version__))
    print("SHA-256 %s" % prov["folds_csv_sha256"])
    print("provenance written to %s" % sidecar)
    insts = sorted({r["instrument"] for r in rows})
    print("fold | images versions | " + " ".join("%4s" % s for s in insts))
    for k, e in folds.fold_table(rows).items():
        print("%4d | %6d %8d | " % (k, e["images"], e["versions"]) + " ".join("%4d" % e["by_instrument"][s] for s in insts))
    print("worst-site image spread %d, fold version-total spread %d"
          % (prov["worst_site_image_spread"], prov["version_total_spread"]))


if __name__ == "__main__":
    main()
