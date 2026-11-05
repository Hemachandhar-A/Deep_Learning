# Solar filament segmentation (GONG H-alpha, Kaggle "Solar Filament Segmentation Challenge 2026")

A reproducible pipeline that segments individual solar filaments in 2048 x 2048 GONG H-alpha images and is scored
with the competition's Panoptic Quality (PQ).  A ResNet34 U-Net (segmentation_models_pytorch) is trained on the mean of
the annotators' masks; filaments are the 8-connected components of the thresholded, upsampled probability map.
The code is written to be read: small modules, each with a docstring that says why.

**Status: Phase 2.1 (Gate 2b) done, waiting for the reviewer.**  The whole pipeline exists and is tested: data,
folds, training (resumable, checksummed), out-of-fold scoring, inference with checked CSV, the Kaggle package and the
end-to-end notebook.  The Kaggle probe and measure sessions have run (nothing was trained); the step budget is **7,000
steps per fold** (`train.steps` in `configs/base.yaml`, approved by the reviewer on 2026-10-15 from the measured
0.2088 s/step).  **No experiment has been run**: the learning-rate check, E1 and E1b wait for the Gate 2b review, the
disc recipe is known to fail on part of the data (`OPEN_QUESTIONS.md` Q1) and every disc option is off.

## Layout

```
src/filament/   rle  pq_metric  data  folds  disc  postprocess  losses  model  train  predict  infer  oof  lrcheck
                config  io_utils  runinfo
scripts/        prepare_cache  make_folds  train  infer  score_oof  lr_check  dry_run_pipeline  session_check
                overlay_check  disc_halo_diagnostic  overfit_test  test5d_reference_run  local_smoke_test  record_environment
                rehearse_kaggle_notebook
configs/        base.yaml, e1_seed1.yaml, e1b_seed2.yaml, e1_lrcheck.yaml (every value cites its document or decision)
tests/          one file per module (pytest); tests that need the competition data are skipped without it
notebooks/      pipeline_source.py -> pipeline.ipynb: preprocessing to final CSV, end to end (TINY switch for a demonstration)
kaggle/         notebook_source.py -> kaggle_notebook.ipynb, kernel-metadata.json, make_dataset.py, prepare_push_folder.py,
                probe_environment.py, RUNBOOK.md
third_party/    Apache-2.0 text for tests/host_scoring_reference.py (see NOTICE)
DECISIONS.md    every non-trivial choice -> document ID -> status
OPEN_QUESTIONS.md  what the documents leave open, the answers, the readings, the incident record
```

## Reproduce (local, Windows or Linux)

```bash
python -m venv venv && venv/Scripts/activate        # or any environment with the versions of requirements.txt
pip install -r requirements.txt                     # segmentation-models-pytorch==0.5.0 downloads the ImageNet weights

export FILAMENT_DATA_ROOT=/path/to/MAGFiLO_1.0_Kaggle_2026   # folder that contains train/ and test/ (read only)
python scripts/prepare_cache.py --data-root $FILAMENT_DATA_ROOT --cache-dir work/cache        # about 2 min
python scripts/make_folds.py    --data-root $FILAMENT_DATA_ROOT --out work/folds.csv           # balanced, seed 42; prints the SHA-256

# the step time is measured with the loader (it prints a derived number and applies nothing):
python scripts/train.py --config configs/e1_seed1.yaml --cache-dir work/cache --folds-csv work/folds.csv \
                        --work-dir work --results-dir results --fold 0 --measure-only 60
# one fold per process (set CUDA_VISIBLE_DEVICES for two cards); the steps come from the config (7,000):
python scripts/train.py --config configs/e1_seed1.yaml ... --fold 0
# --steps N overrides it for a pipeline check only (add --mark-dry: DRY_ run id, recorded in metrics.json)
python scripts/score_oof.py --data-root $FILAMENT_DATA_ROOT --run-dir work/runs/<run_id>   # PQ with bootstrap, grid optional
python scripts/infer.py --config configs/e1_seed1.yaml --run-dir work/runs/<run_id> --cache-dir work/cache --out-dir out

pytest -s                      # unit tests; data tests use FILAMENT_DATA_ROOT and FILAMENT_CACHE_DIR
python scripts/dry_run_pipeline.py --data-root $FILAMENT_DATA_ROOT --work-dir work/dry_run   # tiny end-to-end check
python scripts/session_check.py --expect <commit>      # start and end of every work session (and after any reboot)
```

Nothing is hard-coded: the data are located by searching for `MAGFiLO_1.0_Annotations_kaggle2026_train.json`.
`folds.csv` lists competition file names and is not committed: regenerating it with the same seed gives the same file
(`balanced` strategy, identical under scikit-learn 1.6.1 and 1.9.1); the script writes its SHA-256 and the
scikit-learn version next to it, and every run refuses a file that does not match and records both.  Per-machine
settings (paths, the expected hash) go into a git-ignored `configs/local*.yaml`.  Every result file is written
atomically (temporary file, fsync, rename) and checkpoints carry a checksum, because the development PC reboots
without warning (`OPEN_QUESTIONS.md` Q14).  Kaggle: see `kaggle/RUNBOOK.md`.

## What is measured so far

| What | Value | Where |
| --- | --- | --- |
| Cache build (707 train + 180 test images) | 131 s on a Ryzen 5 5600H, one process | `scripts/prepare_cache.py` |
| ResNet34 U-Net, 512 px, batch 8, mixed precision | 2.16 GiB peak; 0.28 s/step GPU only (RTX 3050 laptop, 4 GiB); 0.309 s/step with the real loader in the main process, 0.285 s with two workers | `scripts/local_smoke_test.py`, `train.py --measure-only` |
| Parameters | 24.43 M | `tests/test_model_loss.py` |
| Whole pipeline on 16 + 8 images, 2 folds, 6 steps | 113 s, every check green (157 s and 144 s in earlier runs) | `scripts/dry_run_pipeline.py` |
| Tesla T4, 512 px, batch 8, AMP, with the loader (Kaggle, 2026-10-14) | 0.2088 s/step (p95 0.2101), 2.17 GiB peak; no slowdown with two processes on two cards | `measure_speed.txt`, `kaggle/probe_environment.py` |
| Step budget | 7,000 steps per fold = 24.4 min on one T4, 2.03 h for five folds | `configs/base.yaml`, `DECISIONS.md` |

A full five-fold training run has not been done yet.

## Data and weights statement

* Only the files the competition provides are used for training; no other labelled solar-filament data (in particular
  not the public MAGFiLO release), and no neighbouring frames at inference.  The model predicts from the single test
  image; flips of that same image are allowed.
* The competition data are licensed CC BY-NC 4.0 and are **not** part of this repository (nor are caches, fold files
  that list image names, checkpoints or predictions).
* Pretrained weights: the ResNet34 ImageNet weights that `segmentation-models-pytorch==0.5.0` downloads from the Hugging
  Face repository `smp-hub/resnet34.imagenet` (revision `7a57b34f723329ff020b3f8bc41771163c519d0c`, pinned inside SMP
  0.5.0; original weights `download.pytorch.org/models/resnet34-333f7ec4.pth`).  That repository's licence field is
  "other"; the competition host accepts a pretrained ResNet34.  Only SMP's CNN encoders are used.
* Rasterisation of annotations uses pycocotools (the host's library), not OpenCV polygon filling.
* `tests/host_scoring_reference.py` is the competition host's scoring code, transcribed from the public "Self
  Evaluation" notebook (Apache License 2.0; header, `NOTICE` and `third_party/LICENSE-Apache-2.0.txt`); it is used only
  to check this repository's PQ replica.
* This code never submits to the competition: it prepares a CSV and checks it; uploading is a human step.

## Licence

Code licence: to be chosen by the team before the repository is made public (OSI-approved, as the rules require).
