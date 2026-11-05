"""Make the folder that ``kaggle kernels push -p <folder>`` uploads: the built notebook and its metadata, nothing else.

    python kaggle/prepare_push_folder.py --out <empty folder>

It builds ``kaggle_notebook.ipynb`` from ``notebook_source.py`` and checks ``kernel-metadata.json`` before copying it:
``enable_gpu`` and ``enable_internet`` are set explicitly (both default to false on Kaggle), the id belongs to the user's
account, the *title resolves to the id* (Kaggle names a kernel after its title; a mismatch makes the second push fail
with HTTP 409), the competition and the code dataset are listed, and no token or password appears anywhere in the folder.
The user, not this script, runs ``kaggle kernels push`` (or the agent, with the user's yes for each push).

``--mode`` sets the parameter lines at the top of the notebook (visible in the built file, nothing else changes).  The
default mode is ``probe``, which trains nothing.

The step budget is ``train.steps`` in ``configs/base.yaml`` (the one place, approved by the reviewer).  A training mode
reads it there: it is refused when that value is ``null``, and ``--steps`` is refused unless it equals the budget or
``--dry`` is given (a pipeline check on the PC: DRY_ run ids, another step count allowed).  ``--folds`` and ``--lr-rows``
select folds and planned learning-rate runs; ``--run-id`` is for ``--mode infer``.  ``--infer-after`` (modes ``e1``,
``e1b`` and ``exp`` only) makes the training session run the test inference itself once all five folds have trained
(Phase 2.2).

``--mode exp --config <name>.yaml`` (Phase 3a) trains the five folds of any experiment config in ``configs/`` (a file name,
not a path): the step budget is that config's ``train.steps`` (which every config inherits from ``configs/base.yaml``) and the
run id comes from the config too.  ``--config`` is refused in every other mode.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
from build_notebook import build  # noqa: E402
from filament.config import load_config  # noqa: E402

MODES = ("probe", "measure", "lrcheck", "e1", "e1b", "exp", "infer")
TRAINING_MODES = ("lrcheck", "e1", "e1b", "exp")
REQUIRED = {"id", "title", "code_file", "language", "kernel_type", "is_private", "enable_gpu", "enable_internet",
            "dataset_sources", "competition_sources", "machine_shape"}
SECRET = re.compile(r"(api[_-]?key|password|passwd|secret|token|KGAT_|\"key\"\s*:)", re.I)


def slug(title: str) -> str:
    """Kaggle's URL slug of a title: lower case, runs of other characters become one hyphen."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def check_metadata(meta: dict) -> list[str]:
    problems = []
    missing = REQUIRED - set(meta)
    if missing:
        problems.append("missing keys: %s" % sorted(missing))
    for k in ("enable_gpu", "enable_internet", "is_private"):
        if k in meta and not isinstance(meta[k], bool):
            problems.append("%s must be a JSON boolean, got %r" % (k, meta[k]))
    if meta.get("competition_sources") != ["filament-segmentation-2026"]:
        problems.append("competition_sources must be ['filament-segmentation-2026']")
    if not str(meta.get("id", "")).startswith("hemachandhara/"):
        problems.append("id must belong to the user's account")
    if meta.get("kernel_type") != "notebook" or not str(meta.get("code_file", "")).endswith(".ipynb"):
        problems.append("a notebook with an .ipynb code file is expected")
    if "id" in meta and "title" in meta and slug(str(meta["title"])) != str(meta["id"]).split("/")[-1]:
        problems.append("the title's slug %r differs from the id's slug %r: Kaggle names the kernel after the title and "
                        "a later push fails with HTTP 409" % (slug(str(meta["title"])), str(meta["id"]).split("/")[-1]))
    return problems


def set_parameters(source: str, mode: str, steps: int | None, folds: list[int] | None, run_id: str | None,
                   dry: bool = False, lr_rows: list[int] | None = None, infer_after: bool = False,
                   config: str | None = None) -> str:
    """Replace the parameter lines at the top of the notebook source; each must be found exactly once."""
    repl = {
        'MODE = "probe"': 'MODE = "%s"' % mode,
        "STEPS = None": "STEPS = %s" % ("None" if steps is None else int(steps)),
        "FOLDS = None": "FOLDS = %s" % ("None" if folds is None else list(folds)),
        "LRCHECK_ROWS = None": "LRCHECK_ROWS = %s" % ("None" if lr_rows is None else list(lr_rows)),
        "DRY = False": "DRY = %s" % bool(dry),
        "INFER_AFTER = False": "INFER_AFTER = %s" % bool(infer_after),
        "CONFIG = None": "CONFIG = %s" % ("None" if config is None else json.dumps(config)),
        "RUN_ID_FOR_INFER = None": "RUN_ID_FOR_INFER = %s" % ("None" if run_id is None else json.dumps(run_id)),
    }
    for old, new in repl.items():
        lines = [l for l in source.splitlines() if l.startswith(old)]
        if len(lines) != 1:
            raise SystemExit("parameter line %r not found exactly once in notebook_source.py" % old)
        source = source.replace(lines[0], new + lines[0][len(old):], 1)
    return source


def approved_budget(config: str | None = None) -> int | None:
    """``train.steps`` as a config resolves it (default: the E1 config; every config inherits base.yaml, the one place)."""
    return load_config(HERE.parent / "configs" / (config or "e1_seed1.yaml"))["train"].get("steps")


def check_budget(mode: str, steps: int | None, dry: bool, budget: int | None) -> list[str]:
    """Why this combination must be refused (empty list = fine)."""
    problems = []
    if steps is not None and steps <= 0:
        problems.append("--steps must be positive")
    if mode in TRAINING_MODES and budget is None:
        problems.append("--mode %s: train.steps is null in configs/base.yaml (no approved step budget): refused (A4.3)" % mode)
    if steps is not None and steps != budget and not dry:
        problems.append("--steps %s differs from the approved budget %s of configs/base.yaml; only a pipeline check "
                        "(--dry) may do that, and the budget changes only with the reviewer's written approval"
                        % (steps, budget))
    if dry and mode not in TRAINING_MODES + ("infer",):
        problems.append("--dry is for the training modes and infer")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mode", choices=MODES, default="probe")
    ap.add_argument("--steps", type=int, default=None, help="override of the budget: only with --dry")
    ap.add_argument("--dry", action="store_true", help="pipeline check: DRY_ run ids, --steps may differ from the budget")
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--lr-rows", type=int, nargs="*", default=None, help="planned lr-check runs to run (default: all)")
    ap.add_argument("--run-id", default=None, help="for --mode infer: the run whose fold models are averaged")
    ap.add_argument("--infer-after", action="store_true",
                    help="e1/e1b/exp: run the test inference in the training session once all five folds have trained")
    ap.add_argument("--config", default=None, help="for --mode exp: the experiment config (a file name in configs/)")
    a = ap.parse_args()
    problems = []
    if a.mode == "exp" and not a.config:
        problems.append("--mode exp needs --config <name>.yaml")
    if a.config is not None and a.mode != "exp":
        problems.append("--config is for --mode exp")
    if a.config is not None and (Path(a.config).name != a.config or not a.config.endswith(".yaml")
                                 or not (HERE.parent / "configs" / a.config).is_file()):
        problems.append("--config %r must be the file name of an existing configs/*.yaml" % a.config)
    if problems:
        raise SystemExit("; ".join(problems))
    budget = approved_budget(a.config)
    problems = check_budget(a.mode, a.steps, a.dry, budget)
    if a.mode == "infer" and not a.run_id:
        problems.append("--mode infer needs --run-id")
    if a.lr_rows is not None and a.mode != "lrcheck":
        problems.append("--lr-rows is for --mode lrcheck")
    if a.infer_after and a.mode not in ("e1", "e1b", "exp"):
        problems.append("--infer-after is for --mode e1, e1b and exp")
    if problems:
        raise SystemExit("; ".join(problems))
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit("%s is not empty" % a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((HERE / "kernel-metadata.json").read_text(encoding="utf-8"))
    problems = check_metadata(meta)
    src_text = set_parameters((HERE / "notebook_source.py").read_text(encoding="utf-8"), a.mode, a.steps, a.folds,
                              a.run_id, dry=a.dry, lr_rows=a.lr_rows, infer_after=a.infer_after, config=a.config)
    tmp_src = a.out / "_source.py"
    tmp_src.write_text(src_text, encoding="utf-8", newline="\n")
    n = build(tmp_src, a.out / meta["code_file"])
    tmp_src.unlink()
    shutil.copy2(HERE / "kernel-metadata.json", a.out / "kernel-metadata.json")
    for f in a.out.iterdir():
        if SECRET.search(f.read_text(encoding="utf-8")):
            problems.append("something that looks like a secret in %s" % f.name)
    if problems:
        raise SystemExit("not ready: %s" % problems)
    print("push folder %s: %s (%d cells) + kernel-metadata.json; checks passed" % (a.out, meta["code_file"], n))
    print("parameters in the notebook: MODE=%r STEPS=%r FOLDS=%r LRCHECK_ROWS=%r DRY=%r INFER_AFTER=%r RUN_ID_FOR_INFER=%r%s"
          % (a.mode, a.steps, a.folds, a.lr_rows, a.dry, a.infer_after, a.run_id,
             " CONFIG=%r" % a.config if a.config is not None else ""))
    if a.mode in TRAINING_MODES:
        print("step budget the session will use: %s per fold (train.steps of configs/base.yaml%s)"
              % (a.steps if a.steps is not None else budget, "; OVERRIDE for a pipeline check" if a.steps is not None
                 and a.steps != budget else ""))
    print("next (the user runs this):  kaggle kernels push -p %s" % a.out)


if __name__ == "__main__":
    main()
