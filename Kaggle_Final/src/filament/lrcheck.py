"""The learning-rate check of experiment E1 (A4.3; protocol Q12, confirmed by the user on 2026-10-13).

The decision rule was written *before* any run, in ``configs/e1_lrcheck.yaml``:

    adopt another learning rate only if its fold-0 out-of-fold PQ exceeds the default's seed-1 PQ by more than the
    absolute difference between the default's seed-1 and seed-2 PQ; otherwise keep the default (1e-4).

This module only applies that rule to numbers; it trains and scores nothing itself.  "Inside the noise" means "not
distinguishable from the default with these two seeds", not "equal".
"""
from __future__ import annotations

from typing import Mapping

DEFAULT_LR = 1e-4


def plan(lr_check_cfg: dict) -> list[dict]:
    """The runs of the protocol as ``[{lr, seed, fold}, ...]`` (from the ``lr_check`` block of the config)."""
    fold = int(lr_check_cfg.get("fold", 0))
    return [{"lr": float(r["lr"]), "seed": int(r["seed"]), "fold": fold} for r in lr_check_cfg["runs"]]


def decide(pq: Mapping[tuple[float, int], float], default_lr: float = DEFAULT_LR) -> dict:
    """Apply the rule.  ``pq[(lr, seed)]`` is the fold-0 PQ (best of the three thresholds) of that run."""
    d1, d2 = pq.get((default_lr, 1)), pq.get((default_lr, 2))
    if d1 is None or d2 is None:
        raise KeyError("the default lr %g needs seeds 1 and 2 (the noise estimate)" % default_lr)
    others = {lr: v for (lr, seed), v in pq.items() if lr != default_lr and seed == 1}
    if not others:
        raise KeyError("no run with another learning rate and seed 1")
    noise = abs(d1 - d2)
    best_lr = max(others, key=lambda k: others[k])
    margin = others[best_lr] - d1
    adopt = margin > noise
    return {
        "default_lr": default_lr, "default_seed1_pq": d1, "default_seed2_pq": d2, "noise_abs_diff": noise,
        "candidates_seed1": {("%g" % k): v for k, v in sorted(others.items())},
        "best_candidate_lr": best_lr, "best_candidate_pq": others[best_lr], "margin_over_default_seed1": margin,
        "adopt_other_lr": bool(adopt), "chosen_lr": best_lr if adopt else default_lr,
        "statement": ("adopt lr %g: its PQ exceeds the default's seed-1 PQ by %.4f, more than the seed noise %.4f"
                      % (best_lr, margin, noise)) if adopt else
                     ("keep lr %g: the best other lr (%g) is %.4f above the default's seed-1 PQ, not more than the "
                      "seed noise %.4f (not distinguishable with two seeds)" % (default_lr, best_lr, margin, noise)),
    }
