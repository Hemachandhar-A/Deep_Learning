"""Score the held-out (out-of-fold) maps of a run (Arch sections 4.5 and 5.3; objective O4).

    python scripts/score_oof.py --data-root <folder> --run-dir <work>/runs/<run_id> [--thresholds 0.3 0.4 0.5]
                                [--grid grid.yaml] [--config cfg.yaml] [--out scores.json] [--n-boot 1000]

No model and no GPU: the maps written by ``train.py`` are upsampled, post-processed with each setting and scored against
every annotator version with the fast scorer.  Prints micro PQ with its image-level bootstrap interval, SQ, RQ, TP, FP,
FN, mean pair Dice, pixel Dice and predicted filaments per image.  The model-comparison setting of A6 is the default:
thresholds 0.30, 0.40, 0.50, no filters, no disc.

``--grid`` is a YAML/JSON mapping of PostprocessConfig keys to lists (threshold, hysteresis, closing_radius, fill_holes,
a_min, c_min); the Cartesian product is scored (E8).  Disc settings (r_max, c_limb) are refused while D-f stands.
Writes ``--out`` atomically.
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import yaml

from filament import data, oof
from filament.config import load_config
from filament.io_utils import atomic_write_json
from filament.postprocess import PostprocessConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=None, help="for the base post-processing settings")
    ap.add_argument("--thresholds", type=float, nargs="*", default=[0.30, 0.40, 0.50])
    ap.add_argument("--grid", type=Path, default=None)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--max-images", type=int, default=None, help="score only the first N held-out images (dry runs)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--allow-disc", action="store_true", help="NOT while the disc recipe is deferred (D-f)")
    a = ap.parse_args()

    ann = data.load_annotations(data.find_data_layout(a.data_root).train_json)
    names, maps = oof.load_oof(a.run_dir, a.folds)
    if a.max_images:
        names, maps = names[: a.max_images], maps[: a.max_images]
    print("held-out maps: %d images, %s, from %s" % (len(names), maps.dtype, a.run_dir.name))

    base = PostprocessConfig.from_dict((load_config(a.config).get("postprocess") if a.config else None) or {})
    if a.grid:
        text = a.grid.read_text(encoding="utf-8")
        grid = json.loads(text) if a.grid.suffix == ".json" else yaml.safe_load(text)
    else:
        grid = {"threshold": a.thresholds}
    settings = oof.grid_settings(base, grid)
    default = PostprocessConfig()  # t = 0.5, no filters: the reference setting of A5.3 / K9
    if default not in settings:
        settings = [default] + settings
    print("scoring %d setting(s) on %d images" % (len(settings), len(names)))
    res = oof.score_settings(maps, names, ann, settings, n_boot=a.n_boot, seed=a.seed, allow_disc=a.allow_disc)

    hdr = "%-52s | %6s %15s | %5s %5s | %5s %5s %5s | %7s %7s | %6s"
    print(hdr % ("setting (non-default values)", "PQ", "95% CI", "SQ", "RQ", "TP", "FP", "FN", "pairDice", "pixDice", "pred/im"))
    for r in res:
        diff = {k: v for k, v in r["params"].items() if v != getattr(PostprocessConfig(), k, None) and k != "out_size"}
        print(hdr % (json.dumps(diff, sort_keys=True)[:52], "%.4f" % r["pq"], "[%.4f, %.4f]" % (r["pq_ci_lo"], r["pq_ci_hi"]),
                     "%.3f" % r["SQ"], "%.3f" % r["RQ"], r["TP"], r["FP"], r["FN"], "%.3f" % r["mean_dice_pairs"],
                     "%.3f" % r["pixel_dice_mean"], "%.1f" % r["pred_per_image_mean"]))
    best = oof.best_setting(res)
    print("best setting: PQ %.4f %s" % (best["pq"], json.dumps({k: v for k, v in best["params"].items()
                                                                 if v != getattr(PostprocessConfig(), k, None)})))
    if a.out:
        atomic_write_json(a.out, {"run": a.run_dir.name, "n_images": len(names), "n_boot": a.n_boot,
                                  "results": oof.jsonable(res)})
        print("wrote", a.out)


if __name__ == "__main__":
    main()
