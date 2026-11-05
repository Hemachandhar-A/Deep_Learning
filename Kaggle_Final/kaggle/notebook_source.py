# %% [markdown]
# # Solar filament segmentation: the Kaggle session
#
# This notebook runs the project's pipeline on Kaggle and prints what each session has to find out.
# **Nothing trains until you change `MODE` below.**  The default `MODE = "probe"` only reads the environment.
#
# * No credentials are used or needed.  Internet must be ON (pip and the ImageNet weights), unless the private code
#   dataset was built with `--with-weights` (then set `OFFLINE_WEIGHTS = True`).
# * It never submits anything to the competition; `infer` only writes a CSV into `/kaggle/working`.
# * The step budget is `train.steps` in `configs/base.yaml` (7,000, approved by the reviewer after `measure`); `STEPS`
#   below is empty and stays empty.  A number there is refused unless `DRY = True` (a pipeline check).
# * Checkpoints are written atomically with checksums and training resumes from them, so a session cut at 12 h loses at
#   most `checkpoint_every` steps (Arch section 4.8 items 6 and 7).
# * A training session ends in an error (not COMPLETE) if any of its processes failed or any step after them failed,
#   after it has written what it has.
# * `INFER_AFTER = True` (modes `e1`, `e1b` and `exp` only): when all five folds have trained, been scored and listed,
#   the same session runs the test-set inference with the five models (no transfer of models) and writes
#   `submission_candidate/` with a second manifest.  Not in a partial session, not after a failure.
# * `MODE = "exp"` (Phase 3a) trains the five folds of the experiment config named in `CONFIG` (a file name in `configs/`,
#   for example `"e1c_seed3.yaml"`), exactly as `e1` does for its config: the step budget and the run id come from that
#   config, then scoring, manifest, listing and, with `INFER_AFTER`, the inference.  The other modes are unchanged.
# * Order of the experiments: `probe`, `measure` (both done), then `lrcheck`, `e1`, `e1b`, `infer`; the reviewer
#   approves each step.

# %%
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

MODE = "probe"  # "probe" | "measure" | "lrcheck" | "e1" | "e1b" | "exp" | "infer"
STEPS = None  # None = the approved budget (train.steps in configs/base.yaml); a number is refused unless DRY = True
FOLDS = None  # folds of this session, e.g. [0, 1]; None = all folds (e1, e1b, infer) or fold 0 (lrcheck)
LRCHECK_ROWS = None  # which planned lr-check runs to run (indexes into the plan of configs/e1_lrcheck.yaml); None = all
DRY = False  # True: a pipeline check (rehearsal on the PC): DRY_ run ids, a STEPS other than the budget is allowed
INFER_AFTER = False  # e1/e1b/exp: after training, scoring and listing, run the test inference in this session (needs all 5 folds)
CONFIG = None  # mode "exp" only: the experiment config in configs/ to train, e.g. "e1c_seed3.yaml"; None in every other mode
TWO_GPU_PROBE = True  # in "probe": also test whether two processes on two cards run side by side at full speed
OFFLINE_WEIGHTS = False  # True: use the ImageNet weights shipped in the code dataset (built with --with-weights)
RUN_ID_FOR_INFER = None  # e.g. "E1_r512_lr1e-4_seed1"; "infer" averages the fold models of that run (FOLDS, else all)

WORK = Path("/kaggle/working")  # the output folder: everything written here is downloadable (limit 20 GB)
TEMP = Path("/kaggle/temp")  # not part of the output; used for the cache if it exists, is writable and has room
INPUT = Path("/kaggle/input")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # the code dataset is read-only
T0 = time.time()
print("MODE =", MODE, "| STEPS =", STEPS, "| FOLDS =", FOLDS, "| LRCHECK_ROWS =", LRCHECK_ROWS, "| DRY =", DRY,
      "| INFER_AFTER =", INFER_AFTER)
assert (MODE == "exp") == (CONFIG is not None), 'CONFIG is set when MODE = "exp" and only then'
assert CONFIG is None or (Path(CONFIG).name == CONFIG and CONFIG.endswith(".yaml")), "CONFIG is a file name in configs/"

# %%
# Dependencies: segmentation_models_pytorch is not preinstalled on Kaggle (verified on 2026-10-12); the version is pinned.
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.5.0"],
                   capture_output=True, text=True)
print("pip install exit code:", r.returncode)
if r.returncode != 0:
    print(r.stderr[-1500:])
import segmentation_models_pytorch as smp  # noqa: E402
import torch  # noqa: E402

print("torch", torch.__version__, "| smp", smp.__version__, "| CUDA", torch.cuda.is_available(),
      "| GPUs", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# %%
# The code: the private dataset made by kaggle/make_dataset.py.  Every file is checked against CODE_MANIFEST.sha256
# before anything from it runs, and the commit is printed.  Kaggle mounts datasets under
# /kaggle/input/datasets/<owner>/<slug>/ and unpacks the zips (verified in the probe of 2026-10-14).
import zipfile


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


manifests = sorted(INPUT.rglob("CODE_MANIFEST.sha256"))  # any depth: do not assume where a dataset is mounted
if not manifests:
    print("folders under", INPUT, "(three levels):")
    for p in sorted(INPUT.rglob("*")):
        if p.is_dir() and len(p.relative_to(INPUT).parts) <= 3:
            print("  ", p)
assert manifests, "no CODE_MANIFEST.sha256 under /kaggle/input (folders listed above): attach the private code dataset"
if len(manifests) > 1:
    print("WARNING: more than one code dataset attached, using", manifests[0], "| others:", manifests[1:])
CODE = manifests[0].parent
# 1. top-level files of the dataset (code.zip, folds.csv, ...): every file that is present must match its checksum
differ, absent = [], []
for line in manifests[0].read_text(encoding="utf-8").splitlines():
    h, rel = line.split("  ", 1)
    p = CODE / rel
    if not p.exists():
        absent.append(rel)  # e.g. a zip that Kaggle unpacked into a folder: covered by the inner manifest below
    elif sha256_of(p) != h:
        differ.append(rel)
assert not differ, "files differ from CODE_MANIFEST.sha256: %s" % differ[:10]
# 2. the code tree: unpacked by Kaggle, or code.zip unpacked here
inner = sorted(CODE.rglob("REPO_MANIFEST.sha256"))
if inner:
    REPO = inner[0].parent
else:
    assert (CODE / "code.zip").exists(), "neither an unpacked code folder nor code.zip in %s" % CODE
    with zipfile.ZipFile(CODE / "code.zip") as z:
        z.extractall(WORK / "code")
    REPO = WORK / "code" / "repo"
# 3. every file of the tree against the inner manifest, before anything from it runs
bad = []
for line in (REPO / "REPO_MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
    h, rel = line.split("  ", 1)
    p = REPO / rel
    if not p.exists() or sha256_of(p) != h:
        bad.append(rel)
assert not bad, "code files differ from REPO_MANIFEST.sha256: %s" % bad[:10]
sys.path.insert(0, str(REPO / "src"))
print("code dataset:", CODE.name, "| commit", (REPO / "COMMIT").read_text().strip(), "| manifests OK",
      "| not present at top level (unpacked by Kaggle):", absent)

# %%
# The approved step budget (one place: configs/base.yaml) and where the cache goes, then what a session has to find out
# (development prompt section 9): data path, Internet, GPUs, persistence, two processes on two cards, the folds hash.
# The weekly quota is read in the Kaggle web page and written in the report.
from filament import data, folds, infer as infer_mod, io_utils, oof, procs, runinfo, train  # noqa: E402
from filament.config import load_config  # noqa: E402

BUDGET = load_config(REPO / "configs" / (CONFIG or "e1_seed1.yaml"))["train"].get("steps")  # exp: from its own config
CACHE, CACHE_WHY = runinfo.choose_cache_dir(WORK, TEMP)  # outside the output folder if /kaggle/temp is usable
print("step budget (configs/base.yaml, train.steps):", BUDGET)
print("cache folder:", CACHE, "|", CACHE_WHY)

folds_src = CODE / "folds.csv"
cmd = [sys.executable, str(REPO / "kaggle" / "probe_environment.py"), "--folds-csv", str(folds_src),
       "--input-root", str(INPUT), "--working", str(WORK), "--out", str(WORK / "probe_report.json"),
       "--cache-dir", str(CACHE), "--temp-root", str(TEMP)]
if BUDGET is not None:
    cmd += ["--step-budget", str(BUDGET)]
if TWO_GPU_PROBE and MODE == "probe":
    cmd.append("--two-gpu")
probe = subprocess.run(cmd, capture_output=True, text=True)
print(probe.stdout)
if probe.returncode != 0:
    print("probe_environment.py exited with", probe.returncode, "\n", probe.stderr[-2000:])

# %%
# Weights: online (default) or the copy in the code dataset.  Kaggle unpacks hf_cache.zip to hf_cache/hf_cache/hub/...:
# the hub folder is searched for, not assumed (finding G2-13).
if OFFLINE_WEIGHTS:
    hub = runinfo.find_hf_hub(CODE)
    if hub is None:
        assert (CODE / "hf_cache.zip").exists(), "the code dataset was built without --with-weights"
        with zipfile.ZipFile(CODE / "hf_cache.zip") as z:
            z.extractall(WORK)
        hub = runinfo.find_hf_hub(WORK / "hf_cache")
    assert hub is not None, "no Hugging Face hub folder (with a models--* folder) in the code dataset"
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ["HF_HUB_OFFLINE"] = "1"
    print("offline weights from", os.environ["HF_HUB_CACHE"])

# The competition data are found by searching for the annotation file name (never a hard-coded path); on Kaggle they
# are under /kaggle/input/competitions/<slug>/ (verified in the probe of 2026-10-14).
hits = list(INPUT.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
assert len(hits) == 1, "expected one annotation file under /kaggle/input, found %s" % hits
layout = data.find_data_layout(hits[0].parents[1])  # <root>/train/<json>: the folder that holds train/ and test/
print("data:", layout.train_json, "|", layout.train_images, "|", layout.test_images)

# %%
# Cache (P8): build once per session (Arch section 4.8 item 2); its time is printed and its checksums recorded.
if not (CACHE / "cache_index.json").exists():
    ann = data.load_annotations(layout.train_json)
    idx = data.build_cache(layout, ann, CACHE)
    print("cache built in %.0f s" % idx["build_seconds"])
print(json.dumps(data.verify_cache(CACHE), indent=1))

# %%
# Folds: copied next to the work folder and verified against the sidecar (the run refuses another file).
FOLDS_CSV = WORK / "folds.csv"
shutil.copy2(folds_src, FOLDS_CSV)
shutil.copy2(CODE / "folds.csv.provenance.json", Path(str(FOLDS_CSV) + ".provenance.json"))
info = folds.verify_for_run(FOLDS_CSV)
print("folds.csv sha256", info["folds_csv_sha256"], "| scikit-learn", info["scikit_learn_version"],
      "| strategy", info["folds_strategy"], "| seed", info["folds_seed"], "| verified", info["folds_verified"])

# %%
# The experiment commands.  One process per fold (CUDA_VISIBLE_DEVICES per card): isolation, resume and two cards.
def effective_steps(config_name):
    """Steps per fold of a training mode: the approved budget of the config.  Another number is a pipeline check only."""
    budget = load_config(REPO / "configs" / config_name)["train"].get("steps")
    if STEPS is not None and STEPS != budget:
        assert DRY, ("STEPS = %s differs from the approved budget %s of configs/base.yaml: only a pipeline check "
                     "(DRY = True) may do that; the budget changes only with the reviewer's written approval"
                     % (STEPS, budget))
    n = STEPS if STEPS is not None else budget
    assert n is not None, "no step budget: train.steps is null in configs/%s (refused, A4.3)" % config_name
    return int(n)


def run_id_for(config_name, lr=None, seed=None):
    """The run id train.py will use (runinfo.make_run_id with the same inputs), to know which runs this session owns."""
    c = load_config(REPO / "configs" / config_name)
    return runinfo.make_run_id(c.get("experiment", "EXP"), int(c["train"].get("resolution", 512)),
                               float(lr if lr is not None else c["optim"]["lr"]),
                               int(seed if seed is not None else c["train"].get("seed", 1)), DRY)


def train_cmd(config, fold, lr=None, seed=None, steps=None):
    c = [sys.executable, str(REPO / "scripts" / "train.py"), "--config", str(REPO / "configs" / config),
         "--cache-dir", str(CACHE), "--work-dir", str(WORK), "--results-dir", str(WORK / "results"),
         "--folds-csv", str(FOLDS_CSV), "--fold", str(fold)]
    if lr is not None:
        c += ["--lr", str(lr)]
    if seed is not None:
        c += ["--seed", str(seed)]
    if steps is not None:
        c += ["--steps", str(steps)]
    if DRY:
        c.append("--mark-dry")
    return c


LISTED = False  # the listing is printed once: by a training session's finish step, else by the last cell


def list_outputs():
    """Names and sizes of everything in the output folder (the cache is listed separately: it is not an output)."""
    global LISTED
    LISTED = True
    tot, n = 0, 0
    for p in sorted(WORK.rglob("*")):
        rel = p.relative_to(WORK)
        if not p.is_file() or rel.parts[0] in ("cache", "code"):
            continue
        print("  %12d  %s" % (p.stat().st_size, rel.as_posix()))
        tot += p.stat().st_size
        n += 1
    print("output folder: %d files, %.1f MB (cache %s: not counted) | session seconds: %.0f"
          % (n, tot / 1e6, "outside the output folder" if WORK not in CACHE.parents else "INSIDE the output folder",
             time.time() - T0))


def infer_after_training(cfgname, rid, results):
    """Option A of the Gate 2b review (Phase 2.2): the test inference in the training session, from the fold models that
    are in /kaggle/working.  Returns the list of problems (empty if it ran cleanly or was legitimately not run)."""
    run_dir = WORK / "runs" / rid
    go, fold_list, why = infer_mod.plan_inference_after_training(dry=DRY, folds_requested=FOLDS, n_folds=5,
                                                                 results=results, run_dir=run_dir)
    print("inference after training:", "RUNNING" if go else "not run", "|", why)
    if not go:
        return []
    problems = []
    try:
        out_dir = WORK / ("DRY_submission_candidate" if DRY else "submission_candidate")
        rep = infer_mod.run_inference(load_config(REPO / "configs" / cfgname), data.load_cache(CACHE), run_dir, out_dir,
                                      "cuda" if torch.cuda.is_available() else "cpu", folds=fold_list, flips=("",))
        print(json.dumps({k: rep[k] for k in ("csv_rows", "images", "csv_check_ok", "round_trip_mismatches",
                                             "components_per_image", "csv_sha256", "folds_used")}, indent=1))
        mf = infer_mod.write_infer_manifest(WORK, out_dir, WORK / "results" / rid / "MANIFEST_infer.sha256")
        print("   manifest:", mf.relative_to(WORK).as_posix())
        print("new files:")
        for q in sorted(list(out_dir.rglob("*")) + [mf]):
            if q.is_file():
                print("  %12d  %s" % (q.stat().st_size, q.relative_to(WORK).as_posix()))
        if not rep["csv_check_ok"] or rep["round_trip_mismatches"]:
            problems.append("inference: the CSV checks failed (csv_check_ok %s, round-trip mismatches %s)"
                            % (rep["csv_check_ok"], rep["round_trip_mismatches"]))
    except Exception:
        traceback.print_exc()
        problems.append("inference after training failed (traceback above)")
    return problems


def finish_training_session(results, what, run_ids, infer=None):
    """After every process returned, in this order: one aggregation of each run's metrics.json (G2-4), out-of-fold scores
    inside the session (R3a), a manifest of SHA-256 values, the listing of the output folder (R3b, R3c), then the
    inference if it was asked for (``infer = (config name, run id)``).  An error in any step is printed with its
    traceback and recorded; nothing already written is removed; the raise comes at the very end (R2, F3)."""
    print(what, "| exit codes:", results)
    problems = []
    ann = None
    for rid in run_ids:
        if not (WORK / "results" / rid / "metrics.json").exists():
            print(rid, ": no fold finished: nothing to aggregate or score")
            continue
        try:
            m = train.aggregate_run_metrics(WORK / "runs" / rid, WORK / "results")
            print(rid, "| folds trained", m["folds_trained"], "| s/step", m.get("seconds_per_step"),
                  "| folds sha256", (m.get("folds_csv_sha256") or "")[:16])
        except Exception:
            traceback.print_exc()
            problems.append("aggregation of %s failed" % rid)
        try:
            if ann is None:
                ann = data.load_annotations(layout.train_json)
            t_score = time.time()
            sc = oof.score_run(WORK / "runs" / rid, ann, n_boot=20 if DRY else 1000, log=None)
            io_utils.atomic_write_json(WORK / "results" / rid / "oof_scores.json", sc)
            for r in sc["results"]:
                print("   t = %.2f | PQ %.4f [%.4f, %.4f] | SQ %.3f RQ %.3f | TP %d FP %d FN %d | filaments per image %.1f"
                      % (r["params"]["threshold"], r["pq"], r["pq_ci_lo"], r["pq_ci_hi"], r["SQ"], r["RQ"], r["TP"],
                         r["FP"], r["FN"], r["pred_per_image_mean"]))
            print("   scoring seconds:", round(time.time() - t_score, 1))
        except FileNotFoundError as e:
            print("   no held-out maps to score:", e)
        except Exception:  # F3: any other error (memory, a bad map) must not skip the manifest and the listing
            traceback.print_exc()
            problems.append("scoring of %s failed" % rid)
        try:
            print("   manifest:", train.write_run_manifest(WORK, rid).relative_to(WORK).as_posix())
        except Exception:
            traceback.print_exc()
            problems.append("manifest of %s failed" % rid)
    list_outputs()
    if infer is not None:
        problems += infer_after_training(infer[0], infer[1], results)
    procs.raise_if_any(results, problems, what)


# %%
N_GPU = max(1, torch.cuda.device_count())
if MODE == "measure":
    # Seconds per step INCLUDING the data loader, on this card, for the E1 baseline; prints the derived step budget.
    cmd = [sys.executable, str(REPO / "scripts" / "train.py"), "--config", str(REPO / "configs" / "e1_seed1.yaml"),
           "--cache-dir", str(CACHE), "--work-dir", str(WORK), "--results-dir", str(WORK / "results"),
           "--folds-csv", str(FOLDS_CSV), "--fold", "0", "--measure-only", "60"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    print(out.stdout[-2500:] or out.stderr[-2500:])
    (WORK / "measure_log.txt").write_text(out.stdout + ("\n--- stderr ---\n" + out.stderr if out.stderr else ""),
                                          encoding="utf-8")
    if out.returncode != 0:  # R2: a failed measurement must not leave a measure_speed.txt or a COMPLETE session
        raise RuntimeError("measure: train.py --measure-only exited with code %d (see measure_log.txt); "
                           "measure_speed.txt was not written" % out.returncode)
    (WORK / "measure_speed.txt").write_text(out.stdout, encoding="utf-8")
elif MODE in ("e1", "e1b"):
    cfgname = "e1_seed1.yaml" if MODE == "e1" else "e1b_seed2.yaml"
    n_steps = effective_steps(cfgname)
    folds_to_run = FOLDS if FOLDS is not None else list(range(5))
    print("%s: %d steps per fold (%s), folds %s, %d process(es) at a time"
          % (MODE, n_steps, "the approved budget" if n_steps == BUDGET else "OVERRIDE: pipeline check", folds_to_run, N_GPU))
    jobs = [("%s fold %d" % (MODE, k), train_cmd(cfgname, k, steps=n_steps)) for k in folds_to_run]
    finish_training_session(procs.run_processes(jobs, N_GPU), MODE, [run_id_for(cfgname)],
                            infer=(cfgname, run_id_for(cfgname)) if INFER_AFTER else None)
elif MODE == "exp":  # Phase 3a: the same steps as e1/e1b for the config named in CONFIG (budget and run id from it)
    n_steps = effective_steps(CONFIG)
    folds_to_run = FOLDS if FOLDS is not None else list(range(5))
    print("exp %s: %d steps per fold (%s), folds %s, %d process(es) at a time"
          % (CONFIG, n_steps, "the approved budget" if n_steps == BUDGET else "OVERRIDE: pipeline check", folds_to_run, N_GPU))
    jobs = [("exp %s fold %d" % (CONFIG, k), train_cmd(CONFIG, k, steps=n_steps)) for k in folds_to_run]
    finish_training_session(procs.run_processes(jobs, N_GPU), "exp " + CONFIG, [run_id_for(CONFIG)],
                            infer=(CONFIG, run_id_for(CONFIG)) if INFER_AFTER else None)
elif MODE == "lrcheck":
    from filament import lrcheck
    if INFER_AFTER:
        print("INFER_AFTER is ignored in mode lrcheck (only e1 and e1b run inference after training)")
    n_steps = effective_steps("e1_lrcheck.yaml")
    plan = lrcheck.plan(load_config(REPO / "configs" / "e1_lrcheck.yaml")["lr_check"])
    rows = list(enumerate(plan)) if LRCHECK_ROWS is None else [(i, plan[i]) for i in LRCHECK_ROWS]
    print("lrcheck: %d steps per run (%s), planned rows %s, %d process(es) at a time"
          % (n_steps, "the approved budget" if n_steps == BUDGET else "OVERRIDE: pipeline check",
             [i for i, _ in rows], N_GPU))
    jobs = [("lrcheck row %d (lr %g, seed %d, fold %d)" % (i, r["lr"], r["seed"], r["fold"]),
             train_cmd("e1_lrcheck.yaml", r["fold"], lr=r["lr"], seed=r["seed"], steps=n_steps)) for i, r in rows]
    finish_training_session(procs.run_processes(jobs, N_GPU), "lrcheck",
                            [run_id_for("e1_lrcheck.yaml", lr=r["lr"], seed=r["seed"]) for _, r in rows])
elif MODE == "infer":
    assert RUN_ID_FOR_INFER, "set RUN_ID_FOR_INFER to the run whose fold models are averaged"
    from filament.infer import run_inference
    cfg = load_config(REPO / "configs" / "e1_seed1.yaml")
    rep = run_inference(cfg, data.load_cache(CACHE), WORK / "runs" / RUN_ID_FOR_INFER,
                        WORK / ("DRY_submission_candidate" if DRY else "submission_candidate"),
                        "cuda" if torch.cuda.is_available() else "cpu", folds=FOLDS, flips=("",))
    print(json.dumps({k: rep[k] for k in ("csv_rows", "images", "csv_check_ok", "round_trip_mismatches",
                                         "components_per_image", "csv_sha256", "folds_used")}, indent=1))
else:
    print("MODE = probe: nothing was trained.")

# %%
# What was written, with names and sizes (the output limit is 20 GB), and the time of the whole session.
if not LISTED:
    list_outputs()
for p in sorted((WORK / "results").glob("*/metrics.json")):
    m = json.loads(p.read_text(encoding="utf-8"))
    print(p.parent.name, "| folds", m["folds_trained"], "| s/step", m.get("seconds_per_step"), "| gpu", m.get("gpu_name"),
          "| folds sha256", (m.get("folds_csv_sha256") or "")[:16], "| steps", m.get("steps"), "| budget", m.get("step_budget"),
          "| cache", m.get("cache_location"))
