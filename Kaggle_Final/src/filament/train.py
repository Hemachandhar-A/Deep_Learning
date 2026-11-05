"""Training of the fold models (Arch sections 4.3, 4.5, 4.8 items 5-7; Gate 1b review B1-B3; Follow-up review P3).

What one fold does
------------------
1. verify ``folds.csv`` against the expected SHA-256 (B1) - done once per run, in ``run_training``;
2. train on the other four folds for a fixed number of *steps* (A4.3), the same number for every fold and arm;
3. predict the held-out fold, whole images at the arm's resolution, and save the maps as float16 (A4.5);
4. keep the *last* model of the schedule, not the best one (A4.3);
5. write ``fold_metrics.json`` and, over all folds present, the run's ``metrics.json``.

Safety against a crash (risk K15, this PC reboots without warning)
-------------------------------------------------------------------
* every file is written atomically (``io_utils``): temporary file, fsync, rename;
* a resume checkpoint is written every ``checkpoint_every`` steps with a ``.json`` sidecar holding its SHA-256, size,
  step and the identity hash of the configuration; on start the newest checkpoint whose checksum matches is loaded and
  training continues from that step with *exactly* the batches it would have seen (the sampler replays its random
  stream); a damaged checkpoint is skipped and reported, a checkpoint of another configuration is refused;
* a fold that already has a valid final model, held-out maps and metrics is not trained again.

The step budget (A4.3) is ``train.steps`` in ``configs/base.yaml``, the one place every training mode reads: 7,000,
approved by the reviewer on 2026-10-15 after ``--measure-only`` gave 0.2088 s per step with the loader on a T4.  ``null``
refuses to train.  ``--steps`` overrides it for a pipeline check only; ``metrics.json`` then says so
(``steps_differs_from_budget``).  ``--measure-only`` prints the derived number and never applies it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import data, folds as folds_mod, io_utils, runinfo
from .config import get as cfg_get, load_with_local, save_config
from .losses import CombinedLoss, LossConfig
from .model import ModelConfig, build_model, count_parameters, pretrained_weights_info
from .predict import predict_probabilities  # noqa: F401  (re-exported: scripts/overfit_test.py imports it from here)


# ---------------------------------------------------------------------------------------------- optimisation
@dataclass(frozen=True)
class OptimConfig:
    optimizer: str = "adam"  # decided 2026-10-12 (D-d): Adam, weight decay 0
    lr: float = 1e-4  # A4.3
    weight_decay: float = 0.0
    warmup_steps: int = 200  # A4.3
    amp: bool = True  # A4.3

    @classmethod
    def from_dict(cls, d: dict | None) -> "OptimConfig":
        return cls(**(d or {}))


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """Multiplier of the base lr at ``step`` (0-based): linear warm-up, then cosine decay to zero at ``total_steps``."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    span = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / span)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def make_optimizer(model: torch.nn.Module, cfg: OptimConfig, total_steps: int):
    if cfg.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError("optimizer must be 'adam' or 'adamw', got %r" % cfg.optimizer)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total_steps, cfg.warmup_steps))
    return opt, sched


def train_loop(model, loader, loss_fn, cfg: OptimConfig, device: str, log_every: int = 25, log=print, *,
               total_steps: int | None = None, start_step: int = 0, state=None, checkpoint_fn=None,
               checkpoint_every: int = 0, history: dict | None = None, step_times: list | None = None) -> dict:
    """Run the steps of ``loader`` (its length is the number of steps still to do).

    ``total_steps`` is the length of the whole schedule (the learning-rate schedule needs it); ``start_step`` is the
    number of steps already done when resuming; ``state`` = (optimizer, scheduler, scaler) to continue from;
    ``checkpoint_fn(done_steps, opt, sched, scaler, history)`` is called every ``checkpoint_every`` steps.
    """
    total = total_steps if total_steps is not None else len(loader) + start_step
    use_amp = bool(cfg.amp and device.startswith("cuda"))
    dev_type = "cuda" if device.startswith("cuda") else "cpu"
    if state is None:
        opt, sched = make_optimizer(model, cfg, total)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        opt, sched, scaler = state
    hist = history if history is not None else {"loss": [], "bce": [], "region": [], "lr": []}
    model.train()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    ckpt_seconds = 0.0
    steps_run = 0
    t_prev = time.perf_counter()
    for i, (x, y) in enumerate(loader):
        done = start_step + i + 1
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        loss, parts = loss_fn(logits, y)  # the loss itself is computed in float32 (losses.py)
        opt.zero_grad(set_to_none=True)
        lr_used = opt.param_groups[0]["lr"]  # the rate of THIS step (G2-6): read before the scheduler moves on
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        hist["loss"].append(float(loss.detach()))
        hist["bce"].append(float(parts["bce"]))
        hist["region"].append(float(parts["region"]))
        hist["lr"].append(lr_used)
        steps_run += 1
        if step_times is not None:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            now = time.perf_counter()
            step_times.append(now - t_prev)
            t_prev = now
        if log and (done % log_every == 0 or done == start_step + 1 or done == total):
            log("  step %5d/%d  loss %.4f (bce %.4f, region %.4f)  lr %.2e  %.1f s"
                % (done, total, hist["loss"][-1], hist["bce"][-1], hist["region"][-1], hist["lr"][-1],
                   time.perf_counter() - t_start))
        if checkpoint_fn is not None and checkpoint_every and done % checkpoint_every == 0 and done < total:
            t0 = time.perf_counter()
            checkpoint_fn(done, opt, sched, scaler, hist)
            ckpt_seconds += time.perf_counter() - t0
            t_prev = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    secs = time.perf_counter() - t_start
    loop_secs = secs - ckpt_seconds
    return {
        "history": hist,
        "seconds": secs,
        "loop_seconds": loop_secs,  # includes the data loader, excludes checkpoint writing
        "checkpoint_seconds": ckpt_seconds,
        "seconds_per_step": loop_secs / max(1, steps_run),
        "steps": total,
        "steps_run": steps_run,
        "amp": use_amp,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else None,
        "state": (opt, sched, scaler),
    }


# ---------------------------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class TrainSettings:
    """Everything that defines how a fold is trained (parsed from the config; the identity of a checkpoint)."""

    experiment: str
    resolution: int
    batch_size: int
    steps: int | None
    seed: int
    crop_min: int | None
    crop_max: int | None
    optim: OptimConfig
    augment: data.AugmentConfig
    model: ModelConfig
    loss: LossConfig
    mean: float
    std: float
    checkpoint_every: int = 500
    num_workers: int = 0
    eval_every: int = 0
    keep_resume_checkpoints: bool = False

    def identity(self, fold: int, folds_sha256: str | None) -> dict:
        d = {
            "resolution": self.resolution, "batch_size": self.batch_size, "steps": self.steps, "seed": self.seed,
            "crop_min": self.crop_min, "crop_max": self.crop_max, "fold": fold, "folds_sha256": folds_sha256,
            "optim": vars(self.optim), "augment": vars(self.augment), "model": vars(self.model),
            "loss": vars(self.loss), "mean": self.mean, "std": self.std,
        }
        return json.loads(json.dumps(d, default=list))  # tuples -> lists, plain JSON types only

    def identity_sha256(self, fold: int, folds_sha256: str | None) -> str:
        return io_utils.canonical_json_sha256(self.identity(fold, folds_sha256))


def settings_from_cfg(cfg: dict, lr: float | None = None, steps: int | None = None, seed: int | None = None,
                      ) -> TrainSettings:
    """Settings from a resolved config; ``lr``, ``steps`` and ``seed`` are the command-line overrides (review B2)."""
    optim = dict(cfg.get("optim") or {})
    if lr is not None:
        optim["lr"] = float(lr)
    aug = dict(cfg.get("augment") or {})
    if aug.get("brightness") is not None:
        aug["brightness"] = tuple(aug["brightness"])
    t = cfg.get("train") or {}
    n = cfg.get("normalisation") or {}
    return TrainSettings(
        experiment=str(cfg.get("experiment", "EXP")),
        resolution=int(t.get("resolution", 512)),
        batch_size=int(t.get("batch_size", 8)),
        steps=int(steps if steps is not None else t["steps"]) if (steps is not None or t.get("steps") is not None) else None,
        seed=int(seed if seed is not None else t.get("seed", 1)),
        crop_min=t.get("crop_min"), crop_max=t.get("crop_max"),
        optim=OptimConfig.from_dict(optim), augment=data.AugmentConfig(**aug),
        model=ModelConfig.from_dict(cfg.get("model")), loss=LossConfig.from_dict(cfg.get("loss")),
        mean=float(n.get("mean", 0.32742)), std=float(n.get("std", 0.22230)),
        checkpoint_every=int(t.get("checkpoint_every", 500)), num_workers=int(t.get("num_workers", 0)),
        eval_every=int(t.get("eval_every", 0)), keep_resume_checkpoints=bool(t.get("keep_resume_checkpoints", False)),
    )


# ---------------------------------------------------------------------------------------------- checkpoints
class CheckpointConfigMismatch(RuntimeError):
    """A checkpoint in the fold folder belongs to a different configuration; it is never silently overwritten."""


def _meta_path(p: Path) -> Path:
    return p.with_name(p.name + ".json")


def save_resume_checkpoint(fdir: Path, step: int, model, opt, sched, scaler, history: dict, config_sha256: str,
                           keep: int = 2) -> Path:
    """Atomic checkpoint of everything needed to continue, then its checksum sidecar, then prune old ones."""
    path = fdir / ("ckpt_step%06d.pt" % step)
    io_utils.atomic_torch_save(path, {
        "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict(), "step": step, "history": history,
    })
    io_utils.atomic_write_json(_meta_path(path), {
        "kind": "resume", "step": step, "bytes": path.stat().st_size, "sha256": io_utils.sha256_file(path),
        "config_sha256": config_sha256,
    })
    old = sorted(fdir.glob("ckpt_step*.pt"))
    for p in old[: max(0, len(old) - keep)]:
        for f in (_meta_path(p), p):
            try:
                f.unlink()
            except OSError:
                pass
    return path


def check_checkpoint(path: Path, expected_config_sha256: str | None = None) -> tuple[bool, str, dict | None]:
    """(valid, reason, meta).  Valid = sidecar present, size and SHA-256 equal to the sidecar, configuration equal."""
    mp = _meta_path(path)
    if not path.exists():
        return False, "file missing", None
    if not mp.exists():
        return False, "checksum sidecar missing", None
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return False, "sidecar unreadable", None
    if path.stat().st_size != meta.get("bytes"):
        return False, "size %d differs from the recorded %s" % (path.stat().st_size, meta.get("bytes")), meta
    if io_utils.sha256_file(path) != meta.get("sha256"):
        return False, "SHA-256 differs from the sidecar (damaged file)", meta
    if expected_config_sha256 is not None and meta.get("config_sha256") != expected_config_sha256:
        return False, "configuration differs", meta
    return True, "ok", meta


def find_resume_checkpoint(fdir: Path, config_sha256: str, log=print):
    """Newest valid resume checkpoint (path, meta) or None.  A damaged one is skipped; another configuration raises."""
    for p in sorted(fdir.glob("ckpt_step*.pt"), reverse=True):
        ok, why, meta = check_checkpoint(p)
        if ok and meta.get("config_sha256") != config_sha256:
            raise CheckpointConfigMismatch(
                "%s was written by another configuration; delete the fold folder or pass --restart" % p.name)
        if ok:
            return p, meta
        if log:
            log("  skipping checkpoint %s: %s" % (p.name, why))
    return None


def load_model_weights(path: str | Path, model: torch.nn.Module, device: str) -> dict:
    """Load a ``model_last.pt`` after checking it against its sidecar (used by inference too)."""
    path = Path(path)
    ok, why, meta = check_checkpoint(path)
    if not ok:
        raise IOError("model file %s is not valid: %s" % (path.name, why))
    sd = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    return meta


# ---------------------------------------------------------------------------------------------- one fold
def holdout_loss(model, images_u8, targets_u8, settings: TrainSettings, device: str, loss_fn, batch_size: int = 4) -> float:
    """Diagnostic only (A4.3): mean loss on the held-out fold.  Never used to choose a checkpoint."""
    model.eval()
    tot, n = 0.0, 0
    use_amp = bool(settings.optim.amp and device.startswith("cuda"))
    with torch.no_grad():
        for i in range(0, len(images_u8), batch_size):
            x = torch.from_numpy(np.stack([data.normalise(np.asarray(a), settings.mean, settings.std)
                                           for a in images_u8[i:i + batch_size]])[:, None]).to(device)
            y = torch.from_numpy(np.stack([np.asarray(a, dtype=np.float32) / 255.0
                                           for a in targets_u8[i:i + batch_size]])[:, None]).to(device)
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.float16, enabled=use_amp):
                z = model(x)
            tot += float(loss_fn(z, y)[0]) * len(x)
            n += len(x)
    model.train()
    return tot / max(1, n)


FOLD_FILES = ("model_last.pt", "model_last.pt.json", "oof_probs.npy", "oof_index.json", "loss_history.json",
              "fold_metrics.json")


def write_fold_manifest(fdir: Path) -> dict:
    """``fold_manifest.json``: SHA-256 and size of every output file of one fold (Phase 2.1, item R3b), so that each
    piece of a session's output can be fetched alone with ``kernels output --file-pattern`` and checked on the PC."""
    files = {}
    for name in FOLD_FILES:
        p = fdir / name
        if p.exists():
            files[name] = {"sha256": io_utils.sha256_file(p), "bytes": p.stat().st_size}
    manifest = {"fold_dir": fdir.name, "files": files}
    io_utils.atomic_write_json(fdir / "fold_manifest.json", manifest)
    return manifest


def fold_is_complete(fdir: Path, config_sha256: str) -> dict | None:
    """The metrics of a fold that finished before (valid final model, maps and metrics), else None."""
    fm = fdir / "fold_metrics.json"
    if not fm.exists():
        return None
    try:
        m = json.loads(fm.read_text(encoding="utf-8"))
    except Exception:
        return None
    ok, _, _ = check_checkpoint(fdir / "model_last.pt", config_sha256)
    oof = fdir / "oof_probs.npy"
    if ok and oof.exists() and io_utils.sha256_file(oof) == m.get("oof_sha256"):
        return m
    return None


def train_fold(settings: TrainSettings, cache: data.Cache, rows: list[dict], fold: int, fdir: Path, device: str,
               folds_sha256: str | None, resume: bool = True, log=print) -> dict:
    """Train one fold and write its outputs into ``fdir``.  Returns the fold's metrics."""
    if settings.steps is None:
        raise ValueError("train.steps is not set: measure the step time with the loader first (--measure-only, A4.3) "
                         "and put the number in the config, or pass --steps for a dry run")
    t_fold = time.perf_counter()
    fdir.mkdir(parents=True, exist_ok=True)
    cfg_sha = settings.identity_sha256(fold, folds_sha256)
    done = fold_is_complete(fdir, cfg_sha)
    if done is not None:
        log("fold %d: already complete (valid model, maps and metrics) - not trained again" % fold)
        return done

    names = cache.train_names
    train_idx, val_idx = folds_mod.fold_indices(rows, names, fold)
    if not train_idx or not val_idx:
        raise ValueError("fold %d has %d training and %d held-out images" % (fold, len(train_idx), len(val_idx)))
    images, targets = cache.arm(settings.resolution)
    side = int(images.shape[-1])
    crop_sides = data.crop_sides_for(side, settings.crop_min, settings.crop_max)
    fold_seed = settings.seed * 1000 + fold  # one stream per (training seed, fold); documented in DECISIONS
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed % (2**32))

    model = build_model(settings.model).to(device)
    loss_fn = CombinedLoss(settings.loss)
    use_amp = bool(settings.optim.amp and device.startswith("cuda"))
    opt, sched = make_optimizer(model, settings.optim, settings.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = {"loss": [], "bce": [], "region": [], "lr": []}
    start_step = 0
    resumed_from = None
    if resume:
        found = find_resume_checkpoint(fdir, cfg_sha, log)
        if found is not None:
            path, meta = found
            ck = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            scaler.load_state_dict(ck["scaler"])
            history = {k: list(v) for k, v in ck["history"].items()}
            start_step = int(ck["step"])
            resumed_from = start_step
            log("fold %d: resuming from %s (step %d of %d)" % (fold, path.name, start_step, settings.steps))
    if start_step >= settings.steps:
        start_step = settings.steps - 1  # a stale resume file at the end: redo the last step rather than skip training

    sampler = data.SpecSampler(train_idx, settings.batch_size, settings.steps, side, crop_sides, settings.augment,
                               seed=fold_seed, start_step=start_step)
    ds = data.FilamentDataset(images, targets, settings.mean, settings.std)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=settings.num_workers,
                                         pin_memory=device.startswith("cuda"))

    def _ckpt(step, o, s, sc, h):
        save_resume_checkpoint(fdir, step, model, o, s, sc, h, cfg_sha)

    log("fold %d: %d training images, %d held-out, %d steps (batch %d, %d px, lr %.1e, seed %d), device %s"
        % (fold, len(train_idx), len(val_idx), settings.steps, settings.batch_size, side, settings.optim.lr,
           settings.seed, device))
    out = train_loop(model, loader, loss_fn, settings.optim, device, log=log, total_steps=settings.steps,
                     start_step=start_step, state=(opt, sched, scaler), checkpoint_fn=_ckpt,
                     checkpoint_every=settings.checkpoint_every, history=history)

    # the last model of the schedule (A4.3), atomically, with its checksum sidecar
    mp = fdir / "model_last.pt"
    io_utils.atomic_torch_save(mp, {"model": model.state_dict()})
    io_utils.atomic_write_json(_meta_path(mp), {
        "kind": "final", "step": settings.steps, "bytes": mp.stat().st_size, "sha256": io_utils.sha256_file(mp),
        "config_sha256": cfg_sha,
    })

    t_pred = time.perf_counter()
    val_images = [images[i] for i in val_idx]
    guard: dict = {}
    probs = predict_probabilities(model, val_images, settings.mean, settings.std, device, amp=settings.optim.amp,
                                  batch_size=4, flips=("",), out_dtype=np.float16, stats=guard)
    io_utils.atomic_save_npy(fdir / "oof_probs.npy", probs)
    io_utils.atomic_write_json(fdir / "oof_index.json", {
        "names": [names[i] for i in val_idx], "cache_indices": [int(i) for i in val_idx], "resolution": side,
        "dtype": "float16", "flips": [""],
    })
    pred_seconds = time.perf_counter() - t_pred

    diag = None
    if settings.eval_every:
        diag = holdout_loss(model, val_images, [targets[i] for i in val_idx], settings, device, loss_fn)

    metrics = {
        "fold": fold, "n_train": len(train_idx), "n_val": len(val_idx), "steps": settings.steps,
        "steps_run": out["steps_run"], "resumed_from_step": resumed_from,
        "seconds_per_step": out["seconds_per_step"], "loop_seconds": out["loop_seconds"],
        "checkpoint_seconds": out["checkpoint_seconds"], "predict_seconds": pred_seconds,
        "nan_guard": {k: int(guard.get(k, 0)) for k in ("passes", "recomputed", "nonfinite_left")},
        "nonfinite_pixels_oof": int(np.count_nonzero(~np.isfinite(probs))),
        "total_seconds": time.perf_counter() - t_fold, "peak_gpu_gib": out["peak_gpu_gib"],
        "final_loss": out["history"]["loss"][-1] if out["history"]["loss"] else None,
        "amp": out["amp"], "n_parameters": count_parameters(model), "fold_seed": fold_seed,
        "config_sha256": cfg_sha, "model_sha256": io_utils.sha256_file(mp),
        "oof_sha256": io_utils.sha256_file(fdir / "oof_probs.npy"), "holdout_loss_diagnostic": diag,
    }
    io_utils.atomic_write_json(fdir / "loss_history.json", out["history"])
    io_utils.atomic_write_json(fdir / "fold_metrics.json", metrics)
    write_fold_manifest(fdir)
    if not settings.keep_resume_checkpoints:
        for p in fdir.glob("ckpt_step*.pt"):
            for f in (_meta_path(p), p):
                try:
                    f.unlink()
                except OSError:
                    pass
    return metrics


# ---------------------------------------------------------------------------------------------- whole run
def publishable_config(cfg: dict) -> dict:
    """The config as it may appear in ``results/`` (review Q13): no machine paths, no file locations."""
    out = json.loads(json.dumps(cfg, default=list))
    out.pop("paths", None)
    if isinstance(out.get("folds"), dict):
        out["folds"].pop("file", None)
    return out


def _fold_metrics_present(run_dir: Path) -> dict:
    out = {}
    for fm in sorted(run_dir.glob("fold_*/fold_metrics.json")):
        m = json.loads(fm.read_text(encoding="utf-8"))
        out[str(m["fold"])] = m
    return out


def aggregate_run_metrics(run_dir: Path, results_dir: Path) -> dict:
    """One aggregation step after all processes of a session returned (Gate 2 finding G2-4).

    Every fold process rewrites ``metrics.json`` from the fold files present at that moment, so two processes that end
    within milliseconds can leave a file that lacks one fold.  This re-reads the fold files once and rewrites only the
    fold-derived keys; everything else in the file is kept."""
    run_dir = Path(run_dir)
    mp = Path(results_dir) / run_dir.name / "metrics.json"
    if not mp.exists():
        raise FileNotFoundError("%s: no metrics.json; no fold of this run has finished" % run_dir.name)
    metrics = json.loads(mp.read_text(encoding="utf-8"))
    fm = _fold_metrics_present(run_dir)
    secs = [m["seconds_per_step"] for m in fm.values() if m.get("seconds_per_step") is not None]
    metrics.update({
        "folds_trained": sorted(int(k) for k in fm), "folds": fm,
        "seconds_per_step": float(np.mean(secs)) if secs else None,
        "total_time_seconds": float(sum(m.get("total_seconds", 0.0) for m in fm.values())),
    })
    io_utils.atomic_write_json(mp, metrics)
    return metrics


def write_run_manifest(work_dir: Path, run_id: str) -> Path:
    """``results/<run_id>/MANIFEST.sha256``: ``<sha256>  <path relative to the work folder>`` for every file of the run
    (fold outputs and results; resume checkpoints and the manifest itself excluded), checkable with ``sha256sum -c``."""
    work_dir = Path(work_dir)
    lines = []
    for sub in (work_dir / "runs" / run_id, work_dir / "results" / run_id):
        for p in sorted(sub.rglob("*")):
            if (not p.is_file() or p.name == "MANIFEST.sha256" or p.name.startswith("ckpt_step")
                    or p.name.endswith(".tmp")):
                continue
            lines.append("%s  %s" % (io_utils.sha256_file(p), p.relative_to(work_dir).as_posix()))
    out = work_dir / "results" / run_id / "MANIFEST.sha256"
    io_utils.atomic_write_text(out, "\n".join(sorted(lines, key=lambda l: l.split("  ", 1)[1])) + "\n")
    return out


def write_run_metrics(run_id: str, cfg: dict, settings: TrainSettings, run_dir: Path, results_dir: Path,
                      folds_info: dict, device: str, dry_run: bool, extras: dict | None = None) -> dict:
    """``results/<run_id>/metrics.json`` over the folds present in ``run_dir`` (idempotent; safe from two processes).

    No absolute path, user name or host name is written (review Q13)."""
    fold_metrics = _fold_metrics_present(run_dir)
    secs = [m["seconds_per_step"] for m in fold_metrics.values() if m.get("seconds_per_step") is not None]
    metrics = {
        "run_id": run_id, "experiment": settings.experiment, "dry_run": dry_run,
        "resolution": settings.resolution, "batch_size": settings.batch_size, "steps": settings.steps,
        "lr": settings.optim.lr, "optimizer": settings.optim.optimizer, "weight_decay": settings.optim.weight_decay,
        "warmup_steps": settings.optim.warmup_steps, "amp": settings.optim.amp, "training_seed": settings.seed,
        "augment": vars(settings.augment), "model": vars(settings.model), "loss": vars(settings.loss),
        "pretrained_weights": pretrained_weights_info(settings.model.encoder, settings.model.encoder_weights),
        **folds_info,
        **runinfo.git_info(), **runinfo.gpu_info(device), "environment": runinfo.environment_info(),
        "folds_trained": sorted(int(k) for k in fold_metrics), "folds": fold_metrics,
        "seconds_per_step": float(np.mean(secs)) if secs else None,
        "total_time_seconds": float(sum(m.get("total_seconds", 0.0) for m in fold_metrics.values())),
    }
    if extras:
        metrics.update(extras)
    io_utils.atomic_write_json(results_dir / run_id / "metrics.json", metrics)
    return metrics


def run_training(cfg: dict, cache: data.Cache, rows: list[dict], folds_info: dict, fold_list: list[int],
                 work_dir: Path, results_dir: Path, device: str, *, lr: float | None = None, steps: int | None = None,
                 seed: int | None = None, resume: bool = True, dry_run: bool = False, extras: dict | None = None,
                 log=print) -> dict:
    """Train ``fold_list`` of one run.  ``folds_info`` is what ``folds.verify_for_run`` returned (or the dry-run stand-in).

    ``extras`` goes into ``metrics.json`` (the CLI passes the cache location); the approved step budget of the config
    and whether this run differs from it are always recorded."""
    settings = settings_from_cfg(cfg, lr=lr, steps=steps, seed=seed)
    budget = (cfg.get("train") or {}).get("steps")
    extras = {**(extras or {}), "step_budget": budget, "steps_differs_from_budget": settings.steps != budget}
    run_id = runinfo.make_run_id(settings.experiment, settings.resolution, settings.optim.lr, settings.seed, dry_run)
    run_dir = work_dir / "runs" / run_id
    out_dir = results_dir / run_id
    save_config({**publishable_config(cfg), "_resolved": {"lr": settings.optim.lr, "steps": settings.steps,
                                                          "training_seed": settings.seed, "run_id": run_id}},
                out_dir / "config.yaml")
    log("run %s: folds %s, device %s%s" % (run_id, fold_list, device, " (DRY RUN)" if dry_run else ""))
    for fold in fold_list:
        train_fold(settings, cache, rows, fold, run_dir / ("fold_%d" % fold), device,
                   folds_info.get("folds_csv_sha256"), resume=resume, log=log)
        write_run_metrics(run_id, cfg, settings, run_dir, results_dir, folds_info, device, dry_run, extras)
    return write_run_metrics(run_id, cfg, settings, run_dir, results_dir, folds_info, device, dry_run, extras)


def measure_speed(cfg: dict, cache: data.Cache, rows: list[dict], fold: int, device: str, n_steps: int = 60,
                  warmup: int = 5, lr: float | None = None, log=print) -> dict:
    """Seconds per step *including the data loader* on the real data (A4.3), and the derived step budget.

    Nothing is trained for real: a fresh model runs ``warmup + n_steps`` steps.  The number it prints is the derivation of
    A4.3 (24 minutes per fold = 1,440 s divided by the measured seconds per step), for the team and the reviewer to
    decide on; it is not written into any config.
    """
    s = settings_from_cfg(cfg, lr=lr, steps=warmup + n_steps)
    names = cache.train_names
    train_idx, _ = folds_mod.fold_indices(rows, names, fold)
    images, targets = cache.arm(s.resolution)
    side = int(images.shape[-1])
    torch.manual_seed(0)
    model = build_model(s.model).to(device)
    loader = torch.utils.data.DataLoader(
        data.FilamentDataset(images, targets, s.mean, s.std),
        batch_sampler=data.SpecSampler(train_idx, s.batch_size, warmup + n_steps, side,
                                       data.crop_sides_for(side, s.crop_min, s.crop_max), s.augment, seed=0),
        num_workers=s.num_workers, pin_memory=device.startswith("cuda"))
    times: list[float] = []
    out = train_loop(model, loader, CombinedLoss(s.loss), s.optim, device, log=None, step_times=times)
    t = np.array(times[warmup:])
    per_step = float(t.mean())
    res = {
        "device": runinfo.gpu_info(device)["gpu_name"], "batch_size": s.batch_size, "resolution": side,
        "measured_steps": int(len(t)), "seconds_per_step_mean": per_step, "seconds_per_step_median": float(np.median(t)),
        "seconds_per_step_p95": float(np.percentile(t, 95)), "steps_per_second": 1.0 / per_step,
        "peak_gpu_gib": out["peak_gpu_gib"],
        "derived_steps_for_24_min_per_fold": int(1440.0 / per_step),
        "derived_note": "A4.3: 24 minutes per fold = 2 h over 5 folds; 1,440 s / measured seconds per step. "
                        "A suggestion for the team and the reviewer, not applied anywhere.",
    }
    log(json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------------------------------------- command line
def _parse(argv):
    ap = argparse.ArgumentParser(prog="python -m filament.train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--local-config", type=Path, default=None, help="git-ignored per-machine overrides (paths, expected hash)")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--folds-csv", type=Path, default=None)
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="folds to train in this process (default: all)")
    ap.add_argument("--fold", type=int, default=None, help="one fold (for one process per GPU)")
    ap.add_argument("--lr", type=float, default=None, help="learning-rate override, recorded in the run id and metrics")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=None, help="data-loader worker processes (default: config, else 0)")
    ap.add_argument("--restart", action="store_true", help="ignore existing resume checkpoints")
    ap.add_argument("--allow-unverified-folds", action="store_true", help="tests and dry runs only")
    ap.add_argument("--dry-run-folds", type=int, default=None,
                    help="dry run on a small cache: deal its images into this many folds (not the real split)")
    ap.add_argument("--measure-only", type=int, default=None, metavar="N", help="measure s/step over N steps and exit")
    ap.add_argument("--mark-dry", action="store_true",
                    help="pipeline check: the run id gets the DRY_ prefix and metrics.json says dry_run (folds are still verified)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = _parse(argv)
    overrides = {"train": {"num_workers": a.num_workers}} if a.num_workers is not None else {}
    cfg = load_with_local(a.config, a.local_config, overrides)
    paths = cfg.get("paths") or {}
    cache_dir = a.cache_dir or paths.get("cache_dir") or (Path(paths["work_dir"]) / "cache" if paths.get("work_dir") else None)
    work_dir = Path(a.work_dir or paths.get("work_dir") or "work")
    results_dir = Path(a.results_dir or paths.get("results_dir") or "results")
    if cache_dir is None:
        raise SystemExit("no cache folder: pass --cache-dir or set paths.cache_dir / paths.work_dir")
    cache = data.load_cache(cache_dir)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    dry_run = a.dry_run_folds is not None
    mark_dry = dry_run or a.mark_dry
    if dry_run:
        versions_of = {n: list(range(v)) for n, v in zip(cache.train_names, cache.n_versions)}
        rows = folds_mod.round_robin_folds(versions_of, a.dry_run_folds)
        folds_info = {"folds_csv_sha256": None, "expected_sha256_source": None, "folds_verified": False,
                      "scikit_learn_version": folds_mod.sklearn.__version__, "folds_strategy": "round_robin_dry_run",
                      "folds_seed": None}
        n_folds = a.dry_run_folds
    else:
        folds_path = a.folds_csv or paths.get("folds_csv") or (cfg.get("folds") or {}).get("file")
        if not folds_path:
            raise SystemExit("no folds file: pass --folds-csv or set folds.file")
        folds_info = folds_mod.verify_for_run(folds_path, (cfg.get("folds") or {}).get("sha256"),
                                              allow_unverified=a.allow_unverified_folds)
        rows = folds_mod.read_folds_csv(folds_path)
        n_folds = int((cfg.get("folds") or {}).get("n_splits", 5))
    problems = folds_mod.check_folds(rows, {n: [0] * v for n, v in zip(cache.train_names, cache.n_versions)}, n_folds) \
        if len(rows) == len(cache.train_names) else ["folds file and cache cover different images"]
    if problems:
        raise SystemExit("folds do not match the cache: %s" % problems)

    fold_list = [a.fold] if a.fold is not None else (a.folds if a.folds else list(range(n_folds)))
    if a.measure_only:
        measure_speed(cfg, cache, rows, fold_list[0], device, n_steps=a.measure_only, lr=a.lr)
        return 0
    run_training(cfg, cache, rows, folds_info, fold_list, work_dir, results_dir, device, lr=a.lr, steps=a.steps,
                 seed=a.seed, resume=not a.restart, dry_run=mark_dry,
                 extras={"cache_location": runinfo.cache_location_label(cache_dir)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
