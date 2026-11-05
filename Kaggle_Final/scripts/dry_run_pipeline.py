"""Dry run of the whole pipeline on a tiny subset, from the raw files to a checked CSV (development prompt section 7, item 4).

    python scripts/dry_run_pipeline.py --data-root <folder> --work-dir <empty folder> [--log run.log]

It executes ``notebooks/pipeline_source.py`` (the text from which ``notebooks/pipeline.ipynb`` is built) with
``FILAMENT_TINY=1``: 16 training images, 8 test images, two folds, six optimiser steps per fold.  Numbers from so few
steps mean nothing; what this shows is that every stage connects and every check is green.  Afterwards it runs drills on the
saved test maps: several thresholds that produce several components per image, an engineered overlap that the checker
must catch, and the id format.  (An image without predictions gets no rows, Spec S4; that behaviour is covered by the
unit tests ``test_image_without_predictions_has_no_rows`` and
``test_maps_to_submission_writes_checked_csv_without_rows_for_empty_images``, not by a drill here.)  Exit code 0 only
if everything passed.

This is a pipeline check, not an experiment: no E1, E1b or learning-rate run happens here.
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    a = ap.parse_args()
    if a.work_dir.exists() and any(a.work_dir.iterdir()):
        raise SystemExit("%s is not empty" % a.work_dir)
    os.environ.update({"FILAMENT_TINY": "1", "FILAMENT_DATA_ROOT": str(a.data_root), "FILAMENT_WORK_DIR": str(a.work_dir)})
    t0 = time.time()
    print("=" * 100)
    print("DRY RUN of notebooks/pipeline_source.py, TINY = 1 (%s)" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 100)
    ns = runpy.run_path(str(REPO / "notebooks" / "pipeline_source.py"), run_name="__main__")

    # ---- drills on the saved test maps -------------------------------------------------------------------------
    from filament import infer, rle
    from filament.postprocess import PostprocessConfig

    print("=" * 100)
    print("DRILLS on the saved test maps")
    out = ns["OUT"]
    maps = np.load(out / "test_probs.npy")
    names = json.loads((out / "test_probs_index.json").read_text(encoding="utf-8"))["names"]
    ok = True
    # thresholds chosen from the maps themselves so that several components per image appear; components below 300 px
    # are dropped, as in the TINY pipeline, because six steps of training leave speckle (thousands of tiny components)
    qs = np.quantile(maps.astype(np.float32), [0.5, 0.9, 0.99])
    for t in sorted({float(round(q, 3)) for q in qs}) + [0.5]:
        rep = infer.maps_to_submission(maps, names, PostprocessConfig(threshold=t, a_min=300), a.work_dir / ("drill_t%.3f.csv" % t),
                                       log=lambda *_: None)
        good = rep["csv_check_ok"] and rep["round_trip_mismatches"] == 0
        ok &= good
        print("  threshold %.3f -> rows %4d, images with rows %d/%d, components per image mean %.1f max %d | checks %s"
              % (t, rep["csv_rows"], rep["images_with_rows"], rep["images"], rep["components_per_image"]["mean"],
                 rep["components_per_image"]["max"], "GREEN" if good else "RED"))
    # engineered overlap: two masks claiming the same 100 pixels must be caught
    m1 = np.zeros((2048, 2048), np.uint8)
    m1[100:200, 100:300] = 1
    m2 = np.zeros((2048, 2048), np.uint8)
    m2[150:250, 250:400] = 1  # overlaps m1 on rows 150-199, cols 250-299 = 50 x 50 = 2500 px
    bad_csv = a.work_dir / "drill_overlap.csv"
    rle.write_submission_csv(bad_csv, [(names[0] + "_1", rle.encode_mask(m1)), (names[0] + "_2", rle.encode_mask(m2))])
    caught = not rle.check_submission_csv(bad_csv, expected_images=names).ok
    print("  engineered overlap caught by the checker:", "GREEN" if caught else "RED")
    ok &= caught
    # an image without predictions has no rows; ids follow <image>_<k>
    rows = rle.read_submission_csv(a.work_dir / ("drill_t%.3f.csv" % 0.5))[1]
    ids_ok = all(r[0].rsplit("_", 1)[0] in names and r[0].rsplit("_", 1)[1].isdigit() for r in rows)
    print("  ids of the t = 0.5 CSV follow <image>_<k> and name test images:", "GREEN" if ids_ok else "RED")
    ok &= ids_ok
    print("=" * 100)
    print("DRY RUN %s in %.0f s" % ("PASSED: every check green" if ok else "FAILED", time.time() - t0))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
