"""Run the Kaggle notebook source on this PC against a fake ``/kaggle`` folder tree (a rehearsal, not an experiment).

    python scripts/rehearse_kaggle_notebook.py --data-root <folder with train/ and test/> \
        --dataset <folder made by kaggle/make_dataset.py> --work <folder> --mode probe
    python scripts/rehearse_kaggle_notebook.py ... --work <same folder> --reuse --mode e1 --folds 0 --steps 6

What it does, in this order:

1. builds ``<work>/kaggle/input/datasets/<owner>/<slug>`` (a copy of the code dataset, three folders deep on purpose),
   ``<work>/kaggle/input/competitions/<slug>`` (a *copy* of the competition files, in the layout the probe found on
   Kaggle on 2026-10-14; the originals are only read), an empty ``<work>/kaggle/working`` and an empty
   ``<work>/kaggle/temp`` (so that the cache goes outside the output folder as it does on Kaggle when the folder is
   usable).  With ``--reuse`` an existing tree is kept, so that several modes share one cache;
2. takes ``kaggle/notebook_source.py``, sets the parameter lines with the same function ``prepare_push_folder.py`` uses,
   replaces the three constants ``/kaggle/input``, ``/kaggle/working`` and ``/kaggle/temp`` by those folders, and runs
   the result as a script in a fresh process (pip, code manifests, probe, data search, cache, folds check, the mode,
   scores, manifest, listing);
3. prints the exit code and the files the notebook wrote.

``probe`` and ``measure`` run as they are.  The training modes (``lrcheck``, ``e1``, ``e1b``, ``exp``) and ``infer`` are only
accepted as a pipeline check: ``DRY = True``, a few steps (``--steps``, default 6), results under ``DRY_`` run ids inside
the fake tree (never in a real work folder).  ``--mode exp --config <name>.yaml`` rehearses any experiment config of
``configs/`` the same way.  The notebook's own exit code is returned, so a drill in which one process
is made to fail (for example ``--folds 0 9``: fold 9 does not exist) ends with a non-zero code.
What this cannot show: Kaggle's own layout beyond what was verified, Internet policy, GPU model, the two-card mapping
(one GPU here), quota.
"""
import _bootstrap  # noqa: F401

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "kaggle"))
from prepare_push_folder import MODES, TRAINING_MODES, set_parameters  # noqa: E402

PIPELINE_CHECK_MODES = TRAINING_MODES + ("infer",)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path, help="folder that contains train/ and test/ (read only)")
    ap.add_argument("--dataset", required=True, type=Path, help="folder made by kaggle/make_dataset.py")
    ap.add_argument("--work", required=True, type=Path, help="folder for the fake /kaggle tree (empty, or --reuse)")
    ap.add_argument("--mode", default="probe", choices=MODES)
    ap.add_argument("--steps", type=int, default=None, help="steps for a training mode (a pipeline check; default 6)")
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--lr-rows", type=int, nargs="*", default=None)
    ap.add_argument("--run-id", default=None, help="for --mode infer")
    ap.add_argument("--infer-after", action="store_true", help="e1/e1b/exp: also rehearse the inference after training")
    ap.add_argument("--config", default=None, help="for --mode exp: the experiment config (a file name in configs/)")
    ap.add_argument("--reuse", action="store_true", help="keep an existing tree (and its cache) from an earlier call")
    ap.add_argument("--test-images", type=int, default=None,
                    help="copy only the first N test images into a NEW tree: a barely trained model makes thousands of "
                         "components per image, so the infer check uses a few images, as the dry run does")
    a = ap.parse_args()
    for need in ("code.zip", "CODE_MANIFEST.sha256", "folds.csv"):
        if not (a.dataset / need).exists():
            raise SystemExit("%s is missing from %s" % (need, a.dataset))
    root = a.work / "kaggle"
    inp, work, temp = root / "input", root / "working", root / "temp"
    if a.reuse:
        if not inp.is_dir():
            raise SystemExit("--reuse: %s has no kaggle/input tree; run once without --reuse" % a.work)
    else:
        if a.work.exists() and any(a.work.iterdir()):
            raise SystemExit("%s is not empty" % a.work)
        shutil.copytree(a.dataset, inp / "datasets" / "hemachandhara" / "solar-filament-code")
        def _keep_first_test_images(folder, names):
            if a.test_images is None or Path(folder).name != "test_images":
                return []
            keep = set(sorted(names)[: a.test_images])
            return [n for n in names if n not in keep]

        shutil.copytree(a.data_root, inp / "competitions" / "filament-segmentation-2026" / a.data_root.name,
                        ignore=_keep_first_test_images)
    work.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=True)

    pipeline_check = a.mode in PIPELINE_CHECK_MODES
    steps = a.steps if a.steps is not None else (6 if a.mode in TRAINING_MODES else None)
    if a.steps is not None and not pipeline_check:
        raise SystemExit("--steps is for the training modes (as a pipeline check)")
    src = (REPO / "kaggle" / "notebook_source.py").read_text(encoding="utf-8")
    if a.infer_after and a.mode not in ("e1", "e1b", "exp"):
        raise SystemExit("--infer-after is for --mode e1, e1b and exp")
    if (a.mode == "exp") != (a.config is not None):
        raise SystemExit("--config is given in --mode exp and only there")
    if a.config is not None and (Path(a.config).name != a.config or not (REPO / "configs" / a.config).is_file()):
        raise SystemExit("--config %r must be the file name of an existing configs/*.yaml" % a.config)
    src = set_parameters(src, a.mode, steps, a.folds, a.run_id, dry=pipeline_check, lr_rows=a.lr_rows,
                         infer_after=a.infer_after, config=a.config)
    for old, new in (('WORK = Path("/kaggle/working")', "WORK = Path(%r)" % str(work)),
                     ('TEMP = Path("/kaggle/temp")', "TEMP = Path(%r)" % str(temp)),
                     ('INPUT = Path("/kaggle/input")', "INPUT = Path(%r)" % str(inp))):
        if src.count(old) != 1:
            raise SystemExit("constant %r not found exactly once in notebook_source.py" % old)
        src = src.replace(old, new)
    script = a.work / ("notebook_rehearsal_%s.py" % (a.mode if a.config is None else "%s_%s" % (a.mode, Path(a.config).stem)))
    script.write_text(src, encoding="utf-8", newline="\n")

    print("rehearsal of kaggle/notebook_source.py, MODE = %s, STEPS = %s, DRY = %s, FOLDS = %s, LRCHECK_ROWS = %s, fake tree %s%s"
          % (a.mode, steps, pipeline_check, a.folds, a.lr_rows, a.work, "" if a.config is None else ", CONFIG = %s" % a.config))
    t0 = time.time()
    code = subprocess.run([sys.executable, "-u", str(script)], cwd=str(a.work)).returncode
    print("-" * 100)
    print("notebook source finished with exit code %d in %.0f s" % (code, time.time() - t0))
    for p in sorted(work.rglob("*")):
        rel = p.relative_to(work)
        if p.is_file() and rel.parts[0] not in ("cache", "code"):
            print("  wrote %-70s %12d bytes" % (rel.as_posix(), p.stat().st_size))
    return code


if __name__ == "__main__":
    sys.exit(main())
