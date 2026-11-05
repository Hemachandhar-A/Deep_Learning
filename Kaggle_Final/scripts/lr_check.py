"""The learning-rate check of E1 (protocol Q12): print the plan, or apply the decision rule to scored runs.

    python scripts/lr_check.py plan   --config configs/e1_lrcheck.yaml
    python scripts/lr_check.py decide --config configs/e1_lrcheck.yaml --data-root <folder> --work-dir <work>

``plan`` prints the five ``scripts/train.py`` commands (fold 0, full E1 budget, which stays ``null`` in the config
until measured: pass ``--steps N`` once it is known).  ``decide`` scores fold 0 of each run's held-out maps at
t in {0.30, 0.40, 0.50} without filters, takes the best t per run, and applies the rule written in advance
(``filament.lrcheck.decide``).  Nothing here trains.
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from filament import data, lrcheck, oof, runinfo
from filament.config import load_config
from filament.io_utils import atomic_write_json
from filament.postprocess import PostprocessConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["plan", "decide"])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--work-dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args()
    cfg = load_config(a.config)
    runs = lrcheck.plan(cfg["lr_check"])
    exp, res = cfg.get("experiment", "E1lr"), int((cfg.get("train") or {}).get("resolution", 512))
    if a.action == "plan":
        for r in runs:
            rid = runinfo.make_run_id(exp, res, r["lr"], r["seed"])
            print("python scripts/train.py --config %s --local-config configs/local.yaml --fold %d --lr %g --seed %d "
                  "--steps <E1 budget>   # run %s" % (a.config, r["fold"], r["lr"], r["seed"], rid))
        return
    ann = data.load_annotations(data.find_data_layout(a.data_root).train_json)
    pq = {}
    for r in runs:
        rid = runinfo.make_run_id(exp, res, r["lr"], r["seed"])
        names, maps = oof.load_oof(a.work_dir / "runs" / rid, [r["fold"]])
        settings = [PostprocessConfig(threshold=t) for t in (0.30, 0.40, 0.50)]
        scored = oof.score_settings(maps, names, ann, settings, n_boot=a.n_boot, log=None)
        best = oof.best_setting(scored)
        pq[(r["lr"], r["seed"])] = best["pq"]
        print("%-28s best t %.2f  fold-%d PQ %.4f  [%.4f, %.4f]" % (rid, best["params"]["threshold"], r["fold"], best["pq"],
                                                                   best["pq_ci_lo"], best["pq_ci_hi"]))
    d = lrcheck.decide(pq)
    print(json.dumps(d, indent=1))
    if a.out:
        atomic_write_json(a.out, d)


if __name__ == "__main__":
    main()
