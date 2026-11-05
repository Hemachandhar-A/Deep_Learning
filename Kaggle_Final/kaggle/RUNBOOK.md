# Kaggle runbook

Nothing in this repository logs in to Kaggle, uploads, or submits.  The commands below use the Kaggle command line tool
with **the user's own credentials**, which the tool reads from the environment (`KAGGLE_API_TOKEN`, a user-level
Windows environment variable) or from `kaggle.json` in the user folder; **no token is written into any file here, and
none may be**.  Account name in the metadata files: `hemachandhara`.  The commands are for Windows PowerShell; `<work>` is
`D:\SOLAR_FIL_2026_kaggle\work`.

Who runs them: the user, or the development agent with the user's yes for every outward action (create a dataset or a
dataset version, push a notebook).  Since 2026-10-14 the agent runs them through a small launcher that is **not part of
this repository** (see section 8).  The agent never submits a CSV to the competition without asking the user first and
getting a clear yes for that one submission; the code itself has no path that submits.

What was learned in the first sessions is in `OPEN_QUESTIONS.md` Q9a; what is still open is Q9.

## 0. Before a session

* Join the competition `filament-segmentation-2026` with the account (done), so that `competition_sources` can attach
  its data to a notebook.
* Have `<work>\folds.csv` and `<work>\folds.csv.provenance.json` (made once by `scripts/make_folds.py`).
* Commit everything: `python scripts/session_check.py --expect <commit>` must say `RESULT: OK`.  The dataset builder
  refuses a tree that differs from HEAD.

## 1. Build the private code dataset (on this PC)

```powershell
python kaggle/make_dataset.py --folds-csv <work>\folds.csv --out <work>\kaggle_dataset --with-weights
```

Expected: `folds.csv verified: sha256 7aec0e67...075f, scikit-learn 1.6.1`, then `files: 5, about 81 MB, commit
<hash>, weights: {...revision 7a57b34f...}`.  The folder holds `code.zip`, `folds.csv`, `folds.csv.provenance.json`,
`hf_cache.zip` (with `--with-weights`), `COMMIT`, `CODE_MANIFEST.sha256`, `dataset-metadata.json`; no data, no caches.
Check the weights fallback once (no Internet needed): `python kaggle/check_offline_weights.py --dataset <work>\kaggle_dataset`
-> `identical encoder weights: True`.
**Kaggle unpacks both zips** (verified on 2026-10-14): `code.zip` appears as `code/repo/...` and `hf_cache.zip` as
`hf_cache/hf_cache/hub/...`; the notebook searches for the folders and checks every file against the manifests.

Create the dataset (private by default; **do not** add `--public`: it contains competition file names), later versions
with `datasets version`:

```powershell
kaggle datasets create  -p <work>\kaggle_dataset
kaggle datasets status  hemachandhara/solar-filament-code        # wait for: ready
kaggle datasets version -p <work>\kaggle_dataset -m "<what changed>"     # a new code version, later
```

Check once on the Kaggle page that the dataset shows **Private** (`isPrivate: true` in `datasets metadata`).

## 2. Push a notebook, watch it, fetch the output

```powershell
python kaggle/prepare_push_folder.py --out <work>\kaggle_push --mode probe        # or measure, lrcheck, e1, e1b, infer
kaggle kernels push   -p <work>\kaggle_push
kaggle kernels status hemachandhara/solar-filament-segmentation-pipeline          # until: COMPLETE (or ERROR)
kaggle kernels output hemachandhara/solar-filament-segmentation-pipeline -p <work>\kaggle_output --file-pattern '^(?!cache/).*'
```

* **The kernel's name is `hemachandhara/solar-filament-segmentation-pipeline`**: Kaggle names a kernel after its title,
  and `kernel-metadata.json` now has that id (a test checks that the title's slug equals the id's slug).  With the earlier
  id `.../solar-filament-pipeline` the second push failed with **HTTP 409 Conflict on SaveKernel** ("Your kernel title does
  not resolve to the specified id"): the title was already taken by the kernel the first push had created under the
  title's slug.  The same name now receives the next version of the same kernel.
* `kernels push` starts a run: GPU on (T4 x2), Internet on, the competition and the code dataset attached
  (`kernel-metadata.json` sets all of these explicitly; `machine_shape: NvidiaTeslaT4` is accepted).
* Fetch with `--file-pattern`: the output folder may hold the cache (1.6 GB; the notebook now puts it in `/kaggle/temp`
  when that folder is usable), and the download of such a file broke twice on this PC (`IncompleteRead`).  The pattern
  above gets every small file; one map: `--file-pattern '^runs/<run_id>/fold_0/oof_probs\.npy$'`.
* After a fetch check what arrived: in `<work>\kaggle_output`, `sha256sum -c results/<run_id>/MANIFEST.sha256` (Git Bash)
  compares every file named in the manifest with its recorded SHA-256 (files that were not fetched are reported missing).

## 3. First two sessions (done): probe and measure

`MODE = "probe"` reads the environment and trains nothing; `MODE = "measure"` runs `train.py --measure-only 60` and writes
`measure_speed.txt` (seconds per step including the loader, p95, peak memory and the derived `1,440 s / s per step`,
applied nowhere).  Results of 2026-10-14: 0.2088 s per step on a T4 with the loader; the reviewer approved 7,000 steps per
fold (`DECISIONS.md`, Phase 2.1).

## 4. Training modes (only after the reviewer has approved Gate 2b)

```powershell
# learning-rate check (protocol Q12; fold 0 only; five runs; configs/e1_lrcheck.yaml)
python kaggle/prepare_push_folder.py --out <work>\p_lr  --mode lrcheck
# E1 and E1b: all folds, or --folds 0 1 for one session, --folds 2 3 4 for the next (resume is automatic)
python kaggle/prepare_push_folder.py --out <work>\p_e1  --mode e1
python kaggle/prepare_push_folder.py --out <work>\p_e1b --mode e1b
# test-set inference from the fold models of a finished run (G2-2: how the models get there is still to be decided)
python kaggle/prepare_push_folder.py --out <work>\p_inf --mode infer --run-id E1_r512_lr1e-3_seed1
```

No `--steps`: the budget is `train.steps` in `configs/base.yaml` (7,000); the builder refuses another number unless
`--dry`, and refuses every training mode if the budget is `null`.  What a training session leaves in `/kaggle/working`:

* `runs/<run_id>/fold_<k>/`: `model_last.pt` (+ `.json` checksum sidecar), `oof_probs.npy` (held-out maps, one file per
  fold), `oof_index.json`, `loss_history.json`, `fold_metrics.json`, `fold_manifest.json`;
* `results/<run_id>/`: `metrics.json` (folds hash, scikit-learn version, commit, lr, seeds, GPU, s/step, total time,
  step budget, cache location, CPU count, library versions), `config.yaml`, **`oof_scores.json`** (out-of-fold PQ with
  bootstrap at t = 0.30, 0.40, 0.50, computed inside the session: the numbers to decide on, a few kilobytes) and
  `MANIFEST.sha256`;
* the last cell prints the name and size of every file.

A session in which any training process failed, or any step after them (scoring, manifest, inference), **ends in an
error** (not COMPLETE) after it has written what it has; the message names each failed process with its exit code and each
later problem, and the traceback is in the log.  The training processes run unbuffered, so their text is expected to reach
the log while they train (not yet seen on Kaggle).

`--infer-after` (modes `e1` and `e1b` only, for the E1 session): when all five folds have trained cleanly, the same session
runs the test inference with the five models (no transfer of models) and leaves `submission_candidate/` (`test_probs.npy`,
`submission.csv`, `infer_report.json`) and `results/<run_id>/MANIFEST_infer.sha256`; if only some folds ran, or a process
failed, it prints why it did not run.  Post-processing can be redone on the PC from `test_probs.npy`.  Whether `/kaggle/working` of one version reaches the next:
no (the cache was rebuilt in version 2), so plan as if each session starts empty and fetch what you need.

## 5. After a session

```powershell
kaggle kernels output hemachandhara/solar-filament-segmentation-pipeline -p <work>\kaggle_output --file-pattern '^(?!cache/).*'
python scripts/score_oof.py --data-root <DATA_ROOT> --run-dir <work>\kaggle_output\runs\<run_id>     # only if the maps were fetched
```

Send back `results/<run_id>/metrics.json` and `oof_scores.json` of each run, `measure_speed.txt` or `probe_report.json`
for those modes, and the kernel log.  The CSV from `infer` mode is only a candidate: the checks passed in the notebook
are repeated on this PC before **the user** uploads it (the agent asks first for every submission).

## 6. Pipeline check on this PC (trains a few steps, discarded; not an experiment)

```powershell
python scripts/rehearse_kaggle_notebook.py --data-root <DATA_ROOT> --dataset <work>\kaggle_dataset --work <work>\rehearsal --mode probe
python scripts/rehearse_kaggle_notebook.py ... --work <work>\rehearsal --reuse --mode e1 --folds 0 --steps 6
```

It runs the notebook source against a fake `/kaggle` tree (code dataset three folders deep, competition under
`competitions/<slug>`, an empty `temp` folder); training modes are always `DRY_` runs of a few steps, and the exit code is
the notebook's: a drill with `--folds 0 9` must end in an error.

## 7. What a session reports about itself

`probe_report.json` (data path, Internet, GPUs and count, CPU count, packages, folds hash, persistence marker, two-GPU
test, step budget, cache location, whether `/kaggle/temp` is usable), the notebook log, `measure_speed.txt`, and from the
Kaggle web page the weekly GPU quota, which no notebook can read.  If a session reports a failed manifest check, the
dataset was altered or truncated: re-create it.

## 8. This PC needs a launcher (not committed)

On the development PC about every second connection that Python's `requests` opens to `api.kaggle.com` is reset
(`ConnectionResetError 10054`; 14 of 30 connections in a test, while `curl` and Python's `http.client` never failed).
Handing `requests` the standard library's default SSL context removes it (0 of 30; certificate and host-name verification
stay on).  The agent therefore runs the tool through `<work>\kaggle_cli.py`, a few lines that do exactly that and then call
the tool's own `main`; it contains no credential and is **kept outside the repository on purpose**.  The plain `kaggle`
command may fail at random on this PC.  Also: `kernels output` of a large file can break mid-transfer; use
`--file-pattern`.
