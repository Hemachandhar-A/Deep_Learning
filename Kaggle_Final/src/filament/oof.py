"""Out-of-fold scoring: saved held-out maps -> PQ against every annotator version (Arch sections 4.5, 5.3, O4).

Post-processing and scoring of the held-out maps need no model and no GPU, so the grid of A5.3 (E8) can be run again and
again on maps written once by ``train.py``.  One image is upsampled once and then passed through every setting of the
grid; masks are scored as RLEs by the fast scorer (``pq_metric``, which equals the host's own functions on 598 pairs).

Reported for every setting: micro PQ with an image-level bootstrap interval (1,000 resamples), SQ, RQ, TP, FP, FN,
mean Dice of matched-or-touching pairs, mean IoU of matched pairs, pixel Dice of the unions, predicted filaments per
image, and the one-to-many / many-to-one counts.  Disc-based settings (``r_max``, ``c_limb``) are refused (D-f).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import data, pq_metric
from .postprocess import PostprocessConfig, postprocess, upsample_probability

GRID_KEYS = ("threshold", "hysteresis", "closing_radius", "fill_holes", "a_min", "c_min", "r_max", "c_limb")


def load_oof(run_dir: str | Path, folds: list[int] | None = None) -> tuple[list[str], np.ndarray]:
    """Held-out maps of the folds present in ``run_dir``: (image names, float16 maps ``(N, H, W)``)."""
    run_dir = Path(run_dir)
    names: list[str] = []
    maps: list[np.ndarray] = []
    fold_dirs = sorted(run_dir.glob("fold_*"), key=lambda p: int(p.name.split("_")[1]))
    for fd in fold_dirs:
        k = int(fd.name.split("_")[1])
        if folds is not None and k not in folds:
            continue
        if not (fd / "oof_probs.npy").exists():
            continue
        idx = json.loads((fd / "oof_index.json").read_text(encoding="utf-8"))
        m = np.load(fd / "oof_probs.npy")
        if m.shape[0] != len(idx["names"]):
            raise ValueError("%s: %d maps but %d names" % (fd.name, m.shape[0], len(idx["names"])))
        names += idx["names"]
        maps.append(m)
    if not maps:
        raise FileNotFoundError("no held-out maps in %s" % run_dir)
    return names, np.concatenate(maps)


def grid_settings(base: PostprocessConfig, grid: dict | None) -> list[PostprocessConfig]:
    """Cartesian product of the lists in ``grid`` (keys of :class:`PostprocessConfig`) on top of ``base``."""
    if not grid:
        return [base]
    bad = [k for k in grid if k not in GRID_KEYS]
    if bad:
        raise ValueError("unknown grid keys %s; allowed: %s" % (bad, GRID_KEYS))
    keys = list(grid)
    out = []
    for combo in itertools.product(*[grid[k] if isinstance(grid[k], (list, tuple)) else [grid[k]] for k in keys]):
        kw = dict(zip(keys, combo))
        if kw.get("hysteresis") is not None:
            kw["hysteresis"] = tuple(kw["hysteresis"])
        out.append(replace(base, **kw))
    return out


def ground_truth(ann: data.Annotations, names) -> tuple[dict, dict]:
    """``({version_id: [RLE]}, {version_id: image})`` for the versions of ``names`` (pycocotools rasterisation, 2048 px)."""
    gt, vi = {}, {}
    for n in names:
        for v in ann.versions_of[n]:
            gt[v] = data.version_rles(ann.version_anns[v])
            vi[v] = n
    return gt, vi


def score_settings(maps, names: list[str], ann: data.Annotations, settings: list[PostprocessConfig],
                   n_boot: int = 1000, seed: int = 0, allow_disc: bool = False, log=print) -> list[dict]:
    """Score every setting on the held-out maps.  Returns one dict per setting (parameters + numbers)."""
    for s in settings:
        if s.needs_disc() and not allow_disc:
            raise ValueError("disc-based settings (r_max / c_limb) are deferred (D-f); got %s" % vars(s))
    gt, vi = ground_truth(ann, names)
    preds = [dict() for _ in settings]  # per setting: image -> list of RLEs
    for i, (name, pm) in enumerate(zip(names, maps)):
        up = upsample_probability(pm, settings[0].out_size)
        for j, s in enumerate(settings):
            preds[j][name] = postprocess(up, s).rles()
        if log and (i + 1) % 100 == 0:
            log("  scored %d / %d images" % (i + 1, len(names)))
    out = []
    for s, pr in zip(settings, preds):
        res = pq_metric.score_fast(gt, pr, vi)
        ci = pq_metric.bootstrap_pq(res, n_boot=n_boot, seed=seed)
        out.append({"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(s).items()},
                    "pq": res.pq, "pq_ci_lo": ci["lo"], "pq_ci_hi": ci["hi"], **{k: v for k, v in res.summary().items()
                                                                                  if k != "PQ"},
                    "result": res})
    return out


def best_setting(results: list[dict]) -> dict:
    return max(results, key=lambda r: r["pq"])


def jsonable(results: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "result"} for r in results]


def score_run(run_dir: str | Path, ann: data.Annotations, thresholds=(0.30, 0.40, 0.50), n_boot: int = 1000, seed: int = 0,
              folds: list[int] | None = None, log=None) -> dict:
    """Score the held-out maps of the folds present in a run folder at the model-comparison settings of A6.

    Thresholds 0.30, 0.40, 0.50, no filters, no disc; the default setting (t = 0.5) is always among them.  This is what
    ``scripts/score_oof.py`` prints, as one small JSON-able dict, so that a Kaggle session can write the numbers the
    reviewer decides on into a file of a few kilobytes (Phase 2.1, item R3a) instead of shipping 70 MB of maps."""
    run_dir = Path(run_dir)
    names, maps = load_oof(run_dir, folds)
    settings = grid_settings(PostprocessConfig(), {"threshold": [float(t) for t in thresholds]})
    default = PostprocessConfig()
    if default not in settings:
        settings = [default] + settings
    res = score_settings(maps, names, ann, settings, n_boot=n_boot, seed=seed, log=log)
    best = best_setting(res)
    present = sorted(int(p.parent.name.split("_")[1]) for p in run_dir.glob("fold_*/oof_probs.npy")
                     if folds is None or int(p.parent.name.split("_")[1]) in folds)
    return {"run": run_dir.name, "folds": present, "n_images": len(names), "n_boot": n_boot, "seed": seed,
            "thresholds": [float(t) for t in thresholds], "results": jsonable(res), "best": jsonable([best])[0]}
