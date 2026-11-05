# %% [markdown]
# # Solar filament segmentation: the whole pipeline, from the raw files to the CSV
#
# Competition: Solar Filament Segmentation Challenge 2026 (GONG H-alpha, MAGFiLO).  This notebook runs every stage with
# the project's own modules (`src/filament/`), each of which says in its docstring what it does and why:
#
# 1. read the annotations, rasterise every filament with pycocotools, average the annotator versions into a soft target;
# 2. cache the images and targets at half resolution (2 x 2 area average);
# 3. a fixed five-fold split, every image with all its annotator versions in one fold;
# 4. train a ResNet34 U-Net per fold (BCE + soft Dice, Adam, cosine schedule, mixed precision), predict the held-out fold;
# 5. score the held-out maps with a replica of the host's Panoptic Quality (checked against the host's own functions);
# 6. average the five fold models (and flips of the same image), take 8-connected components, and write the CSV;
# 7. run every check on the CSV (format, ids, overlaps, round trip).
#
# Rules kept: no extra labels, no neighbouring frames, nothing submitted by this code.
# `TINY = True` runs everything on 16 training and 8 test images, two folds and a handful of steps: it demonstrates that
# the stages connect, its numbers mean nothing.  The full run uses `train.steps` of configs/base.yaml (7,000, the
# approved budget, measured and approved 2026-10-15; see DECISIONS.md).

# %%
import json
import os
import sys
import time
from pathlib import Path

try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:  # inside a notebook
    REPO = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(REPO / "src"))

TINY = os.environ.get("FILAMENT_TINY", "0") == "1"
DATA_ROOT = Path(os.environ["FILAMENT_DATA_ROOT"])  # the folder that holds train/ and test/ (read only)
WORK = Path(os.environ.get("FILAMENT_WORK_DIR", "work_pipeline_tiny" if TINY else "work_pipeline"))
CONFIG = REPO / "configs" / "e1_seed1.yaml"
STEPS = int(os.environ["FILAMENT_STEPS"]) if os.environ.get("FILAMENT_STEPS") else (6 if TINY else None)
T0 = time.time()
print("TINY =", TINY, "| work folder:", WORK.name, "| steps:", STEPS)

# %%
import numpy as np  # noqa: E402
import torch  # noqa: E402

from filament import data, folds, infer, io_utils, oof, rle, runinfo, train  # noqa: E402
from filament.config import load_config  # noqa: E402
from filament.postprocess import PostprocessConfig  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
env = runinfo.environment_info()
print("python", env["python"], "| torch", env["packages"].get("torch"), "| smp", env["packages"].get("segmentation_models_pytorch"),
      "| device", runinfo.gpu_info(device)["gpu_name"])
print("commit", runinfo.git_info(REPO))

# %% [markdown]
# ## 1. Data
# The files are found by searching for the annotation file name; nothing is hard-coded.  Counts are printed so that a
# wrong folder is noticed at once (707 training images, 180 test images, 1,154 annotator versions, 8,199 filaments).

# %%
layout = data.find_data_layout(DATA_ROOT)
ann = data.load_annotations(layout.train_json)
n_test = len(data.list_image_names(layout.test_images))
print("training images %d | annotator versions %d | filaments %d | test images %d"
      % (len(ann.image_names), ann.n_versions, ann.n_filaments, n_test))

# %% [markdown]
# ## 2. Cache
# 1024 px images and soft targets as uint8 (the network trains at 512 px: one more 2 x 2 average).  Soft target = mean of
# the binary masks of the image's annotator versions (rasterised with pycocotools, as the host's scorer does).

# %%
CACHE = WORK / "cache"
if TINY:
    sel_train = data.select_subset(ann.image_names, 16, seed=0)
    sel_test = data.select_subset(data.list_image_names(layout.test_images), 8, seed=0)
    idx = data.build_cache(layout, ann, CACHE, names=sel_train, test_names=sel_test)
elif (CACHE / "cache_index.json").exists():
    idx = json.loads((CACHE / "cache_index.json").read_text(encoding="utf-8"))
else:
    idx = data.build_cache(layout, ann, CACHE)
cache = data.load_cache(CACHE)
print("cache: %d training and %d test images, built in %.0f s" % (len(cache.train_names), len(cache.test_names),
                                                                  idx.get("build_seconds", float("nan"))))

# %% [markdown]
# ## 3. Folds
# Five folds, one row per image, stratified by instrument x number of versions (`StratifiedKFold`, seed 42).  Every
# image, with all its versions, is in exactly one fold, so no held-out score can leak.  `TINY` deals its 16 images into
# two folds in turn instead; that is not the experimental split.

# %%
if TINY:
    versions_of = {n: list(range(v)) for n, v in zip(cache.train_names, cache.n_versions)}
    rows = folds.round_robin_folds(versions_of, 2)
    folds_info = {"folds_csv_sha256": None, "expected_sha256_source": None, "folds_verified": False,
                  "scikit_learn_version": folds.sklearn.__version__, "folds_strategy": "round_robin_dry_run", "folds_seed": None}
    fold_list = [0, 1]
else:
    FOLDS_CSV = WORK / "folds.csv"
    if not FOLDS_CSV.exists():
        rows = folds.make_folds(ann.versions_of)  # defaults: balanced, seed 42, 5 folds
        folds.write_folds_csv(FOLDS_CSV, rows)
        io_utils.atomic_write_json(folds.sidecar_path(FOLDS_CSV), {**folds.provenance(FOLDS_CSV), "strategy": "balanced",
                                                                   "seed": 42, "n_splits": 5, "n_images": len(rows)})
    folds_info = folds.verify_for_run(FOLDS_CSV)
    rows = folds.read_folds_csv(FOLDS_CSV)
    fold_list = list(range(5))
print("folds:", sorted({r["fold"] for r in rows}), "| sha256", (folds_info["folds_csv_sha256"] or "none (dry run)")[:16],
      "| scikit-learn", folds_info["scikit_learn_version"])

# %% [markdown]
# ## 4. Training
# One model per fold on the other four folds, a fixed number of *steps*, the last model of the schedule is kept.  The
# held-out maps are saved as float16.  Checkpoints are atomic and checked on resume.

# %%
cfg = load_config(CONFIG)
if TINY:
    cfg["train"].update(batch_size=4, checkpoint_every=3, num_workers=0)
    # a model trained for six steps produces speckle (thousands of tiny components per image); dropping components below
    # 300 px keeps the demonstration quick.  The real run uses the config's post-processing (no filters, A6).
    cfg["postprocess"].update(a_min=300)
metrics = train.run_training(cfg, cache, rows, folds_info, fold_list, WORK, WORK / "results", device, steps=STEPS,
                             dry_run=TINY)
RUN_ID = metrics["run_id"]
RUN_DIR = WORK / "runs" / RUN_ID
print(RUN_ID, "| folds", metrics["folds_trained"], "| seconds per step (with loader) %.3f" % metrics["seconds_per_step"],
      "| total %.0f s" % metrics["total_time_seconds"])

# %% [markdown]
# ## 5. Out-of-fold score
# Every held-out image is scored against **every** annotator version of that image, as the host's scorer does:
# a hit is IoU > 0.5, PQ = sum of hit IoUs / (TP + FP/2 + FN/2), pooled over all versions.

# %%
names, maps = oof.load_oof(RUN_DIR)
settings = [PostprocessConfig(threshold=t, a_min=300 if TINY else 0) for t in (0.30, 0.40, 0.50)]
scored = oof.score_settings(maps, names, ann, settings, n_boot=50 if TINY else 1000, log=None)
for r in scored:
    print("t = %.2f | PQ %.4f [%.4f, %.4f] | SQ %.3f RQ %.3f | TP %d FP %d FN %d | filaments per image %.1f"
          % (r["params"]["threshold"], r["pq"], r["pq_ci_lo"], r["pq_ci_hi"], r["SQ"], r["RQ"], r["TP"], r["FP"], r["FN"],
             r["pred_per_image_mean"]))
if TINY:
    print("(dry run: 6 steps from the ImageNet weights; these numbers only show that the stages connect)")

# %% [markdown]
# ## 6. Test-set prediction and the CSV
# The five fold models, and flips of the same single test image, are averaged (probabilities), upsampled bilinearly to
# 2048 px, thresholded, split into 8-connected components, holes filled, and written as pycocotools RLE strings.  Every
# model file is checked against its checksum first.  The CSV is only prepared here; uploading is a separate human step.

# %%
OUT = WORK / "submission_candidate"
report = infer.run_inference(cfg, cache, RUN_DIR, OUT, device, folds=fold_list, flips=("", "h"))
print("rows %d | images %d (with rows %d) | filaments per image: mean %.1f, median %.0f (training annotations: median 7)"
      % (report["csv_rows"], report["images"], report["images_with_rows"], report["components_per_image"]["mean"],
         report["components_per_image"]["median"]))

# %% [markdown]
# ## 7. Checks on the CSV
# Header, id format `<image>_<k>`, unique ids, decodable non-empty masks, re-encoding equal, and **no pixel claimed by two
# predictions inside an image** (an overlap would end the upload in ERROR and cost one of the five daily submissions).

# %%
check = rle.check_submission_csv(OUT / "submission.csv", expected_images=cache.test_names)
print(check.summary())
print("round-trip mismatches:", report["round_trip_mismatches"], "| csv sha256", report["csv_sha256"][:16])
assert check.ok and report["round_trip_mismatches"] == 0
print("pipeline finished in %.0f s" % (time.time() - T0))
