# Decisions

Every non-trivial choice in the code, mapped to the document that makes it.  Document IDs: A = Architecture & Pipeline
Decisions (sections, D1-D12, P1-P8, E0-E10, K1-K14), S = Spec & Context; D-a ... D-f are the user's Gate 1 decisions of
2026-10-12 (`4_Gate1_Review.md`, section 4).
Status: **follows document**; **decided** (by the user, with the date); **reading** (an interpretation of ambiguous
text, change on request); **open choice** (documents silent, value picked and marked); **deviation** (differs from the
document; needs the user's approval, none at the moment).  Questions that still need an answer are in
`OPEN_QUESTIONS.md`.

## Model, data, training

| Choice | Value | Document ID | Status |
| --- | --- | --- | --- |
| Network | `smp.Unet('resnet34', imagenet, in_channels=1, classes=1)`, 24.43 M parameters (measured) | D1, A2 | follows document |
| Encoder policy | CNN encoders of SMP only; MiT / SegFormer refused in code | D12, K5 | follows document |
| ImageNet weights | smp 0.5.0 downloads `smp-hub/resnet34.imagenet` at its pinned revision (recorded in each run) | K5 | follows document |
| Input channels | 1; equals three identical channels because SMP sums conv1 weights (tested, max feature difference 1.5e-5) | A2 | follows document, verified |
| Resolution | 2 x 2 area average of 2048 px to 1024 px; the 512 px arm by one more 2 x 2 average | D2, P2, A4.8.1 | follows document |
| Image scaling | divide by 255, minus 0.32742, divide by 0.22230; recomputed over 741,343,232 pixels: 0.32742 / 0.22230 | P1 | follows document, verified |
| Targets | mean of the binary masks of an image's versions, pycocotools rasterisation, 2048 px, then area average | D3, P7 | follows document |
| Version mask | union of the version's filaments | P7 | reading |
| Target storage | `uint8 = round(255 x t)` | P7, P8 | reading |
| Cache | `.npy`, uint8, 1024 px images and targets, test images; built in 131 s on this PC | P8, A4.8.2 | follows document |
| Training sample | 512 px arm: whole image; 1024 px arm: random crop, side a multiple of 32 in [512, 768], one side per batch | A2, A4.8.1 | reading (Q6) |
| Augmentation | **none for E1 and E1b**; flips, 90-degree rotations, brightness x 0.9-1.1 are switches for E6 | A4.4, E6 | decided 2026-10-12 (D-c) |
| Randomness | every random choice of a batch is drawn by `SpecSampler` from one seeded generator | A4.8.5 | implementation |
| Loss | BCE with logits (mean) + soft Dice (batch-summed), weights 1 : 1, smoothing 1 | D5, A4.2 | follows document |
| Focal / Tversky | switchable, off by default; gamma 2; alpha = beta = 0.5 | E4 | open choice |
| Optimiser | **Adam, weight decay 0**, **lr 1e-3** (final: the lr check on fold 0 adopted it, E1b did not reproduce the margin, the lr pair at 1e-4 then gave case 2 of the rule at all three thresholds; see "The lr pair at 1e-4 and the final lr" below) | A4.3 | decided 2026-10-12 (D-d); lr: 1e-3 (review 10), reopened (review 13), 1e-3 final (review 16, 2026-10-18) |
| Learning-rate check (E1) | grid {3e-5, 1e-4, 3e-4, 1e-3} (`lr_check` in `configs/e1_seed1.yaml`, values kept as the record); protocol Q12; run on 2026-10-16/17, outcome 1e-3 (see "lr check result" below), confirmed by the lr pair at 1e-4 (case 2 at all three thresholds, review 16) | A4.3 | decided 2026-10-12 (D-e); outcome accepted 2026-10-17 (review 10), reopened (review 13), settled 2026-10-18 (review 16) |
| Schedule | 200 warm-up steps then cosine to zero; budget counted in steps; AMP on | A4.3 | follows document |
| Step budget | **7,000 steps per fold**, `train.steps` in `configs/base.yaml` (see Phase 2.1) | A4.3 | decided by the reviewer 2026-10-15 |
| Checkpoint reported | last of the fixed schedule | A4.3 | follows document |

## Validation

| Choice | Value | Document ID | Status |
| --- | --- | --- | --- |
| Folds | 5 folds, **one row per image**, label = instrument x number of versions, `StratifiedKFold` (shuffle on): `strategy="balanced"`, the default of `folds.make_folds` and `scripts/make_folds.py`; each image with all its versions in one fold | D9, A4.1 | decided 2026-10-12 (D-a) |
| Fold seed | 42, fixed before any model result exists | A4.1, A4.8.5 | decided 2026-10-12 (D-a) |
| `strategy="document"` | version rows, `StratifiedGroupKFold`: kept only to document the comparison; depends on the scikit-learn version | A4.1 | not used for runs |
| `folds.csv` | written once by `scripts/make_folds.py` into the work folder; **git-ignored**; SHA-256 and scikit-learn version recorded in `folds.csv.provenance.json` and in every run's `metrics.json`; a run refuses a file with another hash; ships in the private Kaggle dataset | A4.1, rule 6 | decided 2026-10-12 (D-b); hash placement Q13 |
| PQ replica | per annotator version, hit cells IoU > 0.5, micro pooling, 0 for an empty denominator; default `matching="host"` | D9, S4 | follows document; confirmed numerically against the host's functions on 598 pairs (review F1) |
| One-to-one matching | `matching="unique"`: the rule of the analysis scripts behind the S5 numbers (TP 2,144); not the competition metric | S5 | kept for comparison |
| Fast scorer | `pycocotools.mask.iou` on RLEs, checked equal to the dense reference and to the host's functions | A4.5 | follows document |
| Bootstrap | image-level, 1,000 resamples, percentile interval, seeded; paired variant for differences | A4.5, A6 | follows document |
| Model comparison score | out-of-fold micro PQ at t in {0.30, 0.40, 0.50}, no filters | A6 | follows document |

## Post-processing and submission

| Choice | Value | Document ID | Status |
| --- | --- | --- | --- |
| Upsampling | bilinear on probabilities (cv2 `INTER_LINEAR`, same convention as torch `align_corners=False`, tested) | A5.2 step 2 | follows document |
| Instances | 8-connected components, holes filled, overlaps resolved after filling | D7, A5.2 | follows document |
| Threshold | strict `p > t` | A5.2 | reading |
| Hysteresis, closing | 8-connected hysteresis; disk kernel of radius rho; both off by default | A5.2, E8 | follows document (hypothesis arms) |
| Filters | a_min, c_min, c_limb as in A5.2 step 7; all off by default | D8 | follows document |
| Disc | recipe of A3 exactly as written; **misfit in part of the data (Q1)**; every disc-based option is off in every config until it is fixed and re-checked, as a separate task before E5 and E8 | A3 | decided 2026-10-12 (D-f): deferred |
| RLE / CSV | counts string only, 2048 x 2048, ids `<image>_<k>`, k from 1; overlap and format checks | S4 | follows document |
| Weights and data | no external labels, no neighbouring frames, no data or caches in the repository | S6, rules 3-6 | follows document |

## Engineering

| Choice | Value | Status |
| --- | --- | --- |
| Rasterisation | pycocotools only (`frPyObjects` + `merge`) | follows document (S5) |
| Image reading | OpenCV `imdecode`, grayscale | reading |
| Paths | `pathlib`; data located by searching for the JSON name; entry points guarded | follows document (A4.8.3, A4.8.9) |
| Host scorer in the tests | `tests/host_scoring_reference.py`, the reviewer's transcription of the public notebook (Apache 2.0), unchanged | review F1; attribution question Q11 |
| Line endings | LF in the repository (`.gitattributes`: `* text=auto eol=lf`) | engineering |
| Local environment | venv layered on the global Python (`--system-site-packages`); versions in `requirements.txt` | engineering |
| `train.py` | schedule, step loop, fold training with verified folds, atomic checksummed checkpoints, exact resume, held-out maps, `metrics.json`, `--measure-only`, one fold per process | Phase 2 |

## Phase 2 (reviews 5 and 6, 2026-10-13)

| Choice | Value | Document ID | Status |
| --- | --- | --- | --- |
| Folds check before a run | `folds.verify_for_run`: expected SHA-256 from the config (`folds.sha256`, git-ignored `configs/local*.yaml`) and/or the sidecar; refuse on mismatch, on disagreement of the two, and when none exists; `--allow-unverified-folds` only for tests and dry runs (a mismatch is refused even then) | B1 | follows review |
| Run record | `metrics.json`: folds hash and source, scikit-learn version, strategy and seed, git commit and dirty flag, lr, optimiser, training seed, GPU name, seconds per step (with loader), total time, environment versions, weights revision, per-fold numbers; no paths, no image names | B1, Q13 | follows review; allowed by the user 2026-10-13 |
| `--lr` override | recorded in the run id (`E1_r512_lr1e-4_seed1`) and in `metrics.json`; the fold is a sub-folder, so two processes can train two folds of one run | B2 | follows review |
| Learning-rate check | `configs/e1_lrcheck.yaml`: fold 0, full E1 schedule, lr 3e-5 / 3e-4 / 1e-3 (seed 1) and 1e-4 (seeds 1, 2); rule fixed in advance (`lrcheck.decide`); step count `null` until measured | Q12, D-e | confirmed by the user 2026-10-13 |
| Step budget | was not set until measured: `--measure-only` prints seconds per step including the loader and the derived `1,440 s / s per step` (A4.3) and applies nothing; set on 2026-10-15, see Phase 2.1 | A4.3 | superseded by Phase 2.1 |
| Checkpoints | temp file in the same folder, flush, fsync, rename; a `.json` sidecar with SHA-256, size, step and the configuration identity; resume takes the newest valid one, skips a damaged one, refuses another configuration; the sampler replays its stream so the resumed run sees the same batches; the last model of the schedule is kept (not the best); completed folds are not retrained | B3, A4.3, A4.8 item 7 | follows review |
| Seeds per fold | `train.seed * 1000 + fold` for the sampler and torch | A4.8 item 5 | reading |
| All result files | written atomically (`io_utils`): folds provenance, metrics, maps, cache files, CSV | P3 | follows review |
| Git | `core.fsync=all` in the repository config; every git call of `session_check.py` with `--no-optional-locks` | P1, P2 | follows review |
| Inference | fold models verified by checksum, flips of the *same* image averaged as probabilities, float16 maps saved, disc options refused, every CSV check run, round trip of every mask | D10, A5.2, D-f | follows document |
| Overlap check | exact in the RLE domain: the union of the masks has the area of the sum of their areas iff no pixel is claimed twice; masks are stacked only to count when that fails | S4 | implementation |
| RLE of components | built from the bounding-box crop, equal to the full-mask encoding (tested); cost does not grow with 2048 x 2048 per component | S4 | implementation (found by the dry run) |
| Out-of-fold scoring | held-out maps upsampled once per image, every setting of the grid applied, fast scorer, bootstrap 1,000; default setting t = 0.5 without filters always included | A5.3, A6 | follows document |
| Host scorer | notice per Q11: header, `third_party/LICENSE-Apache-2.0.txt`, `NOTICE`; code unchanged | Q11 | confirmed by the user 2026-10-13 |
| Code delivery to Kaggle | option B: private dataset with `code.zip` (inner manifest), `folds.csv` + sidecar, optional weights zip; built and used for the probe and measure sessions of 2026-10-14 | Q16 | decided (reviewer, 2026-10-14) |
| Kaggle notebook default | `MODE = "probe"` (reads the environment, trains nothing); training modes read the budget from the config (Phase 2.1); `prepare_push_folder.py --mode ...` sets the parameter lines visibly | B7 | follows review |
| Disc | deferred; settings refused by `infer.py` and `oof.py` | D-f, B7 | follows decision |
| Dry run | `scripts/dry_run_pipeline.py` executes `notebooks/pipeline_source.py` with `TINY=1`; a pipeline check required by the development prompt section 7, not an experiment | prompt 7 | follows prompt |

## Phase 2.1 (`8_Phase2_1_List.md`, 2026-10-15)

| Choice | Value | Document ID | Status |
| --- | --- | --- | --- |
| **Step budget** | **7,000 steps per fold**, the same for every fold and every arm at 512 px, batch 8. Written once: `train.steps` in `configs/base.yaml`, inherited by `e1_seed1`, `e1b_seed2` and `e1_lrcheck` (none sets it again; a test checks this). Measured on a Tesla T4 (Kaggle, 2026-10-14, kernel version 2): 60 steps, one process, with the loader, 512 px, batch 8, AMP: mean 0.2088 s, median 0.2089 s, p95 0.2101 s, peak 2.17 GiB (the probe's benchmark without loader: 0.2123 s). 1,440 s / 0.2088 s = 6,896 (A4.3); 7,000 is that rounded (+1.5%): 7,000 x 0.2088 s = 1,462 s = 24.4 min per fold, 2.03 h for five folds, below A4.3's reference of 7,200. Changes only with the reviewer's written approval. | A4.3 | decided by the reviewer 2026-10-15 |
| Budget refusals | `null` refuses every training mode (`train.py`, notebook, push-folder builder). The notebook's `STEPS` stays `None` (= the config); a number is refused unless `DRY = True` (a pipeline check on the PC); the builder refuses `--steps` other than the budget unless `--dry`; `metrics.json` records `step_budget` and `steps_differs_from_budget` | A4.3, R1 | follows list |
| `num_workers` | stays 0 (the loader costs no visible time on the T4: 0.2088 s with it, 0.2123 s without) | G2-10 | decided with the budget |
| Crashed fold (R2) | `procs.run_processes` returns `(label, exit code)` in job order and prints `GPU n: <command>`; after the metrics, scores, manifest and listing are written the notebook raises `ProcessFailures` naming every failed process; `measure` writes `measure_log.txt` always, `measure_speed.txt` only on success, and raises on a non-zero code | R2, G2-14 | follows list |
| Scoring inside the session (R3a) | after training, `oof.score_run` writes `results/<run_id>/oof_scores.json`: out-of-fold PQ with bootstrap (1,000 resamples; 20 in a DRY run), SQ, RQ, TP, FP, FN at t = 0.30, 0.40, 0.50 (the default t = 0.5 is among them), no filters, no disc: a few kilobytes. `lrcheck.decide` is not run in the notebook and its rule is untouched | R3a | follows list |
| Output layout (R3b, R3c) | per fold `runs/<run_id>/fold_<k>/`: `model_last.pt` (+ `.json` sidecar), `oof_probs.npy` (one file per fold), `oof_index.json`, `loss_history.json`, `fold_metrics.json`, `fold_manifest.json` (SHA-256 and size of the others); per run `results/<run_id>/`: `metrics.json`, `config.yaml`, `oof_scores.json`, `MANIFEST.sha256` (`<sha256>  <path relative to /kaggle/working>`, for `sha256sum -c`); the last cell prints names and sizes of everything in the output folder. Each piece is fetched with `--file-pattern`. | R3b, R3c | proposal for review at Gate 2b |
| `metrics.json` after a session (G2-4) | `train.aggregate_run_metrics` re-reads the fold files once after all processes returned and rewrites only the fold-derived keys | G2-4 | follows list |
| Learning rate in log and `loss_history.json` (G2-6) | the rate of the step that was taken (read before `sched.step()`); the schedule itself was and is unchanged | G2-6 | follows list |
| Directory `fsync` (G2-3) | an `OSError` of the directory `fsync` (or of opening the directory) is ignored; the file `fsync` and the rename stay strict | G2-3 | follows list |
| Cache location | `/kaggle/temp/cache` if `/kaggle/temp` exists, a file can be written in it and it has at least 3 GiB free, else `/kaggle/working/cache`; nothing is assumed (`runinfo.choose_cache_dir`); the choice and its reason are printed and the label (`/kaggle/temp/cache`, `/kaggle/working/cache` or `other`) is in `metrics.json` and the probe report. Not verified on Kaggle that `/kaggle/temp` exists: the probe report now says | list | follows list |
| Offline weights (G2-13) | the `hub` folder below the dataset (`hf_cache/hf_cache/hub` on Kaggle) is searched for, `runinfo.find_hf_hub`; tested with Kaggle's nested layout and the flat one; `OFFLINE_WEIGHTS` stays `False` | G2-13 | follows list |
| Kernel id | `kaggle/kernel-metadata.json` id = `hemachandhara/solar-filament-segmentation-pipeline` (title unchanged); the builder refuses metadata whose title slug differs from the id slug (the 409 of 2026-10-15); the push-folder file has SHA-256 `e23105ce...0235` | list | follows list |
| Run record | `metrics.json` and the probe report carry CPU count, GPU count and names, library versions (torch, numpy, pillow, OpenCV, smp, timm, scikit-learn, huggingface_hub), the step budget and the cache location | list | follows list |
| Pipeline-check mode | notebook parameters `DRY` (DRY_ run ids, `--mark-dry`, another step count allowed) and `LRCHECK_ROWS` (which planned lr runs); `scripts/rehearse_kaggle_notebook.py` runs every mode against a fake `/kaggle` tree; results stay inside that tree | R1 | follows list |
| `infer` with fewer than five folds | `FOLDS` selects the fold models averaged (default all); needed for the rehearsal, no change for the real run | R1 | implementation |
| G2-2 (fold models for `infer`) | **proposal only, not implemented**: see the Gate 2b report, section 5 | G2-2 | waiting for the reviewer |
| G2-5 (reuse of lr-check fold-0 runs) | **decided: no reuse**; E1 and E1b train all five folds; nothing implemented (see "lr check result" below) | G2-5 | decided by the reviewer 2026-10-17 (review 10) |
| Closed notes | G2-9 the reviewer compares the folds hash in every `metrics.json`; G2-10 `num_workers` 0; G2-11 torch 2.11.0+cu128 ran probe and measure, the training branches have not run on it; G2-12 Internet is on and `pip install` works | G2-9 to G2-12 | closed by the reviewer |
| Open for the user | the three real image names in tracked text, the code licence, the commit identity: **not acted on** | review 7 | waiting for the user |

## Phase 2.2 (Gate 2b review `9_Gate2b_Review.md`, section 5.2 and item 4 of the reviewer's message; 2026-10-16)

| Choice | Value | Document ID | Status |
| --- | --- | --- | --- |
| **Inference in the training session (option A)** | notebook parameter `INFER_AFTER` (default `False`), honoured in modes `e1` and `e1b` only (`prepare_push_folder.py --infer-after`). When all five folds have trained cleanly the same session loads the five models from `/kaggle/working`, runs the test inference with `flips = ("",)` (no flip averaging until an experiment says so) and writes `submission_candidate/` exactly as `run_inference` does today (`test_probs.npy`, `submission.csv`, `infer_report.json`, every CSV check and the round trip of every mask) | G2-2, review 9 section 5.2 | approved by the reviewer |
| Order in the session | training, aggregation, scores, run manifest, listing (as before); **then** inference; then `results/<run_id>/MANIFEST_infer.sha256` (SHA-256 of the new files, for `sha256sum -c`), a listing of the new files; only at the very end a raise if anything failed | review 9 5.2 item 3 | follows review |
| When it runs (`infer.plan_inference_after_training`) | real session: only if every process returned 0, the session was asked for all five folds, and each has `model_last.pt`; otherwise it prints why it did not run (a partial session of folds 0 and 1 trains but does not infer, and that alone is not an error). DRY (pipeline check): any subset of folds that trained cleanly, so that the path can be rehearsed on one fold; run folder `DRY_submission_candidate/` | review 9 5.2 item 2 | **reading R18** (`OPEN_QUESTIONS.md`) |
| Errors after the processes returned (F3) | scoring, aggregation, manifest and inference are each wrapped: the traceback is printed, the problem is recorded, the remaining steps still run, nothing already written is removed or hidden; `procs.raise_if_any` raises once at the very end, naming the failed processes **and** the problems | F3 | follows review |
| Unbuffered children | every training process gets `PYTHONUNBUFFERED=1` from `procs.run_processes` (the drill's child text reached the log only when the processes exited) | reviewer's message, item 4 | follows review |
| Training code | `train.py`, the configs, `data.py` and the folds are **byte-identical** to commit `98cd5f2` (checked with `git diff 98cd5f2 -- ...`, see the Gate 2c report); changed or new: `infer.py` (planning function and manifest writer), `procs.py`, the notebook, the builder, the rehearsal harness | review 9 5.2 item 6 | follows review |
| Printed hint of `make_dataset.py` | names `datasets create` (first time) and `datasets version -m "<message>"` (later) | F6 | follows review |
| B (private models dataset), C (kernel as source) | B only for the arm chosen for the final submission; C not now: **not implemented** | review 9 5.2 | decided by the reviewer |

## lr check result (`10_Gate2c_and_lrcheck_Review.md`, 2026-10-17)

*Note added with review 13 (same evening): this section records what was decided and applied at the time (E1 and E1b ran at 1e-3).
The row "Applies to" no longer holds: E1b did not reproduce the margin and the lr is open again; see the next section.  Settled by review 16: lr 1e-3, see the last section.*

Session: Kaggle kernel version 4, code commit `98cd5f2`, fold 0, 7,000 steps per run, 5,789 s; all five manifests checked
(`sha256sum -c`, 50 of 50 OK); `folds_csv_sha256` `7aec0e67...075f` in all five `metrics.json`.  Out-of-fold PQ of fold 0
(142 images) at t = 0.50:

| run | PQ | SQ | RQ | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| lr 3e-5, seed 1 | 0.2048 | 0.625 | 0.328 | 719 | 2024 | 924 |
| lr 1e-4, seed 1 | 0.2151 | 0.630 | 0.341 | 799 | 2241 | 844 |
| lr 1e-4, seed 2 | 0.2134 | 0.630 | 0.339 | 794 | 2251 | 849 |
| lr 3e-4, seed 1 | 0.2033 | 0.625 | 0.325 | 773 | 2340 | 870 |
| **lr 1e-3, seed 1** | **0.2392** | **0.633** | **0.378** | **835** | **1942** | **808** |

| Choice | Value | Document ID | Status |
| --- | --- | --- | --- |
| Rule | written before the runs (Q12, `configs/e1_lrcheck.yaml`, `lrcheck.decide`): adopt another lr only if its fold-0 PQ exceeds the default's seed-1 PQ by more than the absolute seed-1/seed-2 difference of the default | Q12, D-e | confirmed by the user 2026-10-13 |
| Outcome | **adopt 1e-3**: margin over the default's seed 1 0.0240 against a seed-pair difference of 0.0018 at t = 0.50; the same choice at t = 0.30 (0.0148 against 0.0070), t = 0.40 (0.0218 against 0.0020) and best of the three t per run (0.0212 against 0.0027) | review 10 4.1 | accepted by the reviewer 2026-10-17 |
| Applies to | E1, E1b and every config that inherits `configs/base.yaml` (`optim.lr` is the only value that changed; `lr_check` in `e1_seed1.yaml`, `lrcheck.py` `DEFAULT_LR` and `e1_lrcheck.yaml` are the record of the finished check and keep 1e-4 as its reference) | review 10 6 | approved in writing |
| G2-5 | **no reuse** of the fold-0 lr-check runs: E1 and E1b train all five folds (the winner of four candidates is a selected result; E1 is the baseline of every later comparison and should come from one session, one commit and one dataset version) | review 10 5 | decided by the reviewer |
| Grid not extended | no run at 3e-3 or above before E1; extending after seeing the edge is a new experiment | review 10 4.3 | decided by the reviewer |
| Caveats (kept so nobody has to rediscover them) | (1) the noise number 0.0018 comes from one pair of runs; (2) the 3e-4 run is 0.0119 below the 1e-4 run, 6.7 times that pair difference, so either the response to lr is not smooth or the seed-to-seed noise is larger than the pair suggests; (3) 1e-3 is the largest value of the grid, nothing is known above it; (4) one fold and 142 images, the 95 % bootstrap intervals overlap and describe image sampling, not seed noise. **The learning rate was tuned on one fold with one seed and may not be the best; the final report says so** | review 10 4.2 | recorded |
| Check when E1 and E1b return | E1 fold 0 and lrcheck row 4 use the same random stream (`fold_seed = seed * 1000 + fold`, `train.py`), so they differ by GPU effects only; the reviewer compares E1 and E1b fold-0 PQ at t = 0.50 with 0.2392 and with 0.2151 / 0.2134; if E1b fold 0 is not above 0.2151 the advantage did not reproduce at a second seed and a decision is asked for before E2.  This is the reviewer's check, not a rule in the code | review 10 4.4 | recorded |

## E1 and E1b at 1e-3, and the lr pair at 1e-4 (`13_E1b_Review.md`, 2026-10-17)

Both sessions ran on commit `901f28c` (kernel versions 5 and 6, dataset version 3, lr 1e-3, 7,000 steps per fold); every manifest
checked, both candidate CSVs passed the reviewer's check.  Out-of-fold PQ at t = 0.50, per fold and pooled over the 707 images:

| fold | 0 | 1 | 2 | 3 | 4 | pooled (t = 0.30 / 0.40 / 0.50) |
| --- | --- | --- | --- | --- | --- | --- |
| E1, seed 1 | 0.2392 | 0.2015 | 0.2247 | 0.2133 | 0.2075 | 0.2181 / 0.2219 / **0.2165** |
| E1b, seed 2 | 0.2150 | 0.2249 | 0.2253 | 0.2256 | 0.2149 | 0.2229 / 0.2275 / **0.2212** |
| E1b - E1 | -0.0242 | +0.0234 | +0.0006 | +0.0122 | +0.0073 | +0.0048 / +0.0056 / +0.0047 |

Fold-0 PQ at t = 0.50 of every run on the same 142 images: lr 1e-4 seed 1 0.2151, seed 2 0.2134; lr 1e-3 seed 1 0.2392 (the
lr-check row 4 and E1 fold 0 are bit-identical, same model hash), seed 2 0.2150; lr 3e-5 0.2048 and lr 3e-4 0.2033 (seed 1 only).

| Choice | Value | Document ID | Status |
| --- | --- | --- | --- |
| Tripwire (written before E1b) | E1b fold 0 PQ at t = 0.50 had to be above 0.2151 (1e-4, seed 1, same fold).  Result: **0.214986 against 0.215138, 0.000152 below: fired**.  Nothing else was pushed | review 10 4.4, review 12 | recorded |
| What it shows | the two seeds at lr 1e-3 differ by 0.0242 on the same fold and the same images, as much as the margin (0.0240) on which 1e-3 was adopted and about 14 times the noise estimate of the rule (0.0018, one pair of seeds at 1e-4).  Paired by seed, 1e-3 minus 1e-4 is +0.0240 (seed 1) and +0.0016 (seed 2).  **"lr 1e-3 is better than 1e-4" is not established; lr 1e-3 against 1e-4 is open.**  Whether the seed noise at 1e-4 is really as small as 0.0018 is not known | review 13 3 | recorded |
| Noise of the 1e-3 baseline | the five per-fold differences between the two seeds have mean +0.0039 and standard deviation s3 = 0.0177; the pooled gap of the two seeds, 0.0047, is smaller than the typical size of a pooled difference (s3 / sqrt(5) = 0.0079) | review 13 3 | recorded |
| The lr pair at 1e-4 | E1 and E1b again with `optim.lr` 1.0e-4 in `configs/base.yaml` (run ids `E1_r512_lr1e-4_seed1`, `E1b_r512_lr1e-4_seed2`; about 93 min each, about 3.2 h of quota in all).  Order, each outward step after the user's own yes: this commit (local) -> dataset version 4 -> push E1 -> package to the reviewer -> push E1b -> package.  If the first 1e-4 run comes back more than 0.015 below A (below), the reviewer is told before the second is pushed.  E1 and E1b at 1e-3 stay on record | review 13 4 | recommended by the reviewer |
| Rule, written before any 1e-4 result exists | A = mean of the pooled PQ at t = 0.50 of E1 and E1b at 1e-3 = (0.2165 + 0.2212) / 2 = **0.2189**; B = the same for the two 1e-4 runs; s3 = 0.0177; s4 = the standard deviation of the five per-fold differences between the two 1e-4 seeds.  (1) B > A + 0.005: lr 1e-4 for E2 to E9.  (2) B < A - 0.005: lr 1e-3 for E2 to E9.  (3) abs(B - A) <= 0.005: lr 1e-4 only if s4 < 0.5 * s3 (variance at least four times smaller), otherwise lr 1e-3 (no change to make).  The margin 0.005 is about one typical difference of two means of two runs (0.0056, derived), a practical tie-break and not a proof.  Either way, the four sets of out-of-fold maps are kept and their average is used for the ensemble check | review 13 4 | written by the reviewer before the result |
| Noise rule of Architecture section 6 | restated after the pair (the reviewer writes the numbers into the Architecture document); E2 to E9 are on hold until the lr is settled and the rule is restated | review 13 1, 5 | pending |
| Candidate CSVs | E1 (sha256 `2952608753c64ae71c565213405fc7d44c12a9cf40d32997f4de5c28f08c3499`) may be uploaded by the user, their own decision; the E1b CSV is not to be submitted | review 12, 13 | recorded |

## The lr pair at 1e-4 and the final lr (`16_E1b_lr1e-4_Pair_Decision_Review.md`, 2026-10-18)

E1 and E1b were run again at lr 1e-4 (kernel versions 8 and 10, commit `9eddb7b`, dataset version 4, 7,000 steps, seeds 1 and 2; versions 7 and 9 queued
without starting and were cancelled by the user).  Out-of-fold PQ over the 707 images (1,000 bootstrap draws, seed 0):

| run | pooled t=0.30 | t=0.40 | t=0.50 | per fold at t=0.50 (folds 0 to 4) |
| --- | --- | --- | --- | --- |
| E1 at 1e-3, seed 1 | 0.2181 | 0.2219 | 0.2165 | 0.2392 0.2015 0.2247 0.2133 0.2075 |
| E1b at 1e-3, seed 2 | 0.2229 | 0.2275 | 0.2212 | 0.2150 0.2249 0.2253 0.2256 0.2149 |
| E1 at 1e-4, seed 1 | 0.2111 | 0.2128 | 0.2061 | 0.2151 0.2014 0.2070 0.2013 0.2068 |
| E1b at 1e-4, seed 2 | 0.2143 | 0.2161 | 0.2091 | 0.2134 0.2014 0.2074 0.2163 0.2067 |

The rule of review 13 (A = mean of the 1e-3 pair, B = mean of the 1e-4 pair, s3 and s4 = standard deviations of the five per-fold differences between the two seeds):

| t | A | B | B - A | s3 | s4 | case |
| --- | --- | --- | --- | --- | --- | --- |
| 0.30 | 0.2205 | 0.2127 | -0.0078 | 0.0167 | 0.0074 | 2 |
| 0.40 | 0.2247 | 0.2145 | -0.0102 | 0.0164 | 0.0053 | 2 |
| 0.50 | 0.2189 | 0.2076 | -0.0112 | 0.0177 | 0.0069 | 2 |

| Choice | Value | Document ID | Status |
| --- | --- | --- | --- |
| Learning rate for E2 to E9 | **1e-3** (case 2: B < A - 0.005 at all three thresholds, so nothing is decided separately).  `configs/base.yaml` `optim.lr` 1.0e-3 again.  At t = 0.50, lr 1e-3 is above lr 1e-4 in 10 of 10 paired fold comparisons (same seed, same fold).  Limit: 1e-3 is the top of the grid and 3e-3 was never tried; it can be an arm later if quota is left | review 16 3 | decided by the reviewer |
| Integrity | fold 0 of E1 at 1e-3, of E1 at 1e-4 and of E1b at 1e-4 is bit-identical to lr-check rows 4, 0 and 1 (same model hash, same out-of-fold map hash, same config hash), although commit, session and speed differ.  Not verified: that this holds on every card | review 16 2 | recorded |
| Noise of the 1e-3 baseline | s3 = 0.0177 (0.0167 at 0.30, 0.0164 at 0.40); the typical size of a pooled difference between two single runs is s3 / sqrt(5) = 0.0079; the noise of one run is 0.0079 / sqrt(2) = 0.0056.  With four degrees of freedom the 95 % range of the typical difference is 0.005 to 0.023.  The seed noise at 1e-4 looks smaller (s4 = 0.0069); five numbers each, hypothesis only | review 16 3, 6 | recorded |
| Noise rule of Architecture section 6, restated (replaces its rule (b)) | Baseline for E2 to E9: E1 and E1b at lr 1e-3.  An arm is judged on its pooled out-of-fold PQ at its best t among 0.30, 0.40, 0.50, against the baseline pair's mean at its best t (0.2247 at 0.40).  Stage 1 (screening, seed 1): (a) a paired image-bootstrap interval of its PQ difference to E1 above zero and (b) a pooled gain over the baseline mean above one typical difference, 0.0079; otherwise the arm is dropped (ties go to the simpler arm).  Stage 2 (seed 2): adopted only if the mean of its two runs is above the baseline mean by more than 0.0112 (2 sigma) and both runs are individually above the baseline mean.  The rule uses an uncertain sigma: when E1c (a third baseline seed) exists, s3 and sigma are recomputed from three runs and the user is told if they move materially.  Arms are adopted one at a time and the combination re-measured.  The rule judges single models; the final submission is built from an ensemble | review 16 6 | decided by the reviewer |
| Averages of out-of-fold maps (tool `avg_oof_pq.py`, N runs, outside the repository) | pooled PQ at t = 0.30 / 0.40 / 0.50: average of the two at 1e-3 0.2136 / 0.2214 / 0.2460; of the two at 1e-4 0.2167 / 0.2228 / 0.2351; of all four 0.2532 / 0.2551 / 0.2600 (above all seven entries in 5 of 5 folds at 0.50; +0.0389 over the best single run).  Two seeds gain about +0.027 over their members at either lr.  The out-of-fold numbers hold one model per image; the test CSV averages five fold models | review 14, 16 5 | recorded |
| Threshold sweeps 0.20 to 0.90 (information only; 100 bootstrap draws, reduced) | best pooled threshold: single runs 0.40 (E1 0.2219, E1b 0.2275); average of the two at 1e-3 0.55 (0.2539); average of the two at 1e-4 0.50 (0.2351); average of all four 0.50 (0.2600).  The best threshold depends on what is averaged, so a setting tuned on single-run maps does not transfer to the ensemble.  Every best number is picked on the images that score it and is a little inflated.  No rule, CSV or threshold changed | review 14, 16 | recorded |
| The step of the averaged maps near 0.5 (local checks L2) | single runs have a flat histogram between 0.40 and 0.60 (about 5 pixels per million per bin); the average of the two at 1e-3 has about 50,000 pixels in each of the bins 0.49 to 0.50 and 0.50 to 0.51.  Of the 100,526 pixels with an averaged value in 0.49 to 0.51, 99.2 % are pixels where one run is below 0.1 and the other above 0.9: confident disagreement averages to 0.5, so a threshold below 0.5 keeps what either run marks and one above 0.5 keeps what both mark.  The hypothesis that the models predict 0.5 where the target level is 1/2 is not supported by the single-run histograms.  Training targets at 512 px: about half of the marked pixels have in-between values (edges blended by the two downscalings) | review 16 5, local checks | recorded |
| Leaderboard (public board, about half of the test images), reported by the user, two decimals | E1 at 1e-3 0.21; E1b at 1e-3 0.22; E1 at 1e-4 0.19; E1b at 1e-4 0.19.  Same order as the out-of-fold PQ (0.2165, 0.2212, 0.2061, 0.2091).  Whether averaging five fold models helped on the test images is not shown (the board gives two decimals and has a sampling noise of roughly 0.02) | review 16 4 | recorded |
| Finding N1: NaN under mixed precision | the fold-1 model of E1b at 1e-3 gives NaN pixels when run with AMP (fp16 autocast): 9,025 in one out-of-fold image (20131130192134Ch, top-left 95 x 95 px) and 7,505 in each of two test images (20140925195834Ch, 20200310113230Ch, rows 0-78, columns 0-94 of 512); in float32 the same model gives none.  Reproduced on the PC (RTX 3050) with the same pixel counts as on Kaggle (T4).  The other three runs have none.  NaN is never above a threshold, so it counts as background and no check noticed it; re-predicting the one out-of-fold image in float32 changes E1b's pooled PQ at 0.50 by -0.00006.  Open: a NaN check in the training session (train.py is protected) and float32 fallback; the reviewer decides | local checks | open (OPEN_QUESTIONS Q18) |
| Quota pairs | before / after: 30 h to 28 h 26 m (94 min, session 93.0), 28 h 26 m to 26 h 50 m (96 min, 96.8), 26 h 50 m to 25 h 17 m (93 min, 92.3), 25 h 17 m to 23 h 45 m (92 min, 92.7): wall-clock time of one session, readings by the user; cancelled, never-started versions used about nothing | review 15, 16 | recorded |
| Queue stalls | version 7 (104 min) and version 9 (7 min) stayed QUEUED and were cancelled on the Kaggle website; the same files started within 33 s when pushed again.  Cause not known.  Rule relaxed by the reviewer: after about 5 minutes in QUEUED the user may cancel on the website and the agent re-pushes the same files after re-checking the two hashes, writing down all times and statuses; nothing that has started to run is cancelled | review 16 8 | decided by the reviewer |
| E2 to E9 and their code | config-only by code reading: E6 (augmentation switches exist in the sampler), E4 focal (`loss.bce: focal`) and E4 Tversky (`loss.region: tversky`), E7 (`model.arch`, `encoder`, `encoder_weights`; only the ResNet34 weights are shipped in the code dataset, other encoders need Internet or a larger dataset), E2 (`train.resolution: 1024`, `crop_min`, `crop_max`; needs a cost measurement; its out-of-fold maps are 1.5 GB per run).  Need code: E3 (only the soft mean is implemented; union and majority could be derived from the stored soft target, one sample per version cannot), E5 (no limb flattening or CLAHE in the training code, and the disc recipe is deferred, D-f).  E8 and E9 are CPU work on saved maps | Phase 3a | recorded |
| E4 loss parameters | Architecture A4.2 names the arms ("focal loss; Tversky loss") and gives no values; the code defaults come from `losses.py` ("open choice").  Focal gamma 2.0 is the usual value and the code default.  Tversky with alpha = beta = 0.5 (the code default) is the Dice loss up to the smoothing constant (they differ by about 1e-6 on a batch, checked), so it is not an arm; `configs/e4b_tversky.yaml` carries a PROPOSAL (alpha 0.7 on false positives, beta 0.3 on false negatives), not from the documents, to be confirmed or changed by the reviewer before any run | A4.2, Phase 3a | open (OPEN_QUESTIONS Q17) |
| Generic training mode `exp` (Phase 3a) | notebook `MODE = "exp"` with `CONFIG = "<name>.yaml"`: trains the five folds of that config, scores, lists, supports INFER_AFTER; run id and step budget come from the config; `prepare_push_folder.py --mode exp --config <name>.yaml`; the modes probe, measure, lrcheck, e1, e1b and infer are unchanged.  Arm configs on base.yaml, each differing from E1 only in the keys it names (a test): `e1c_seed3`, `e6a_flip_rot90`, `e6b_flip_rot90_brightness`, `e4a_focal_dice`, `e4b_tversky` | review 16 7 | built, awaiting review |
