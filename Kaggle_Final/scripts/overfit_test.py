"""Overfit eight training images (Arch section 6, experiment E0; acceptance test 11).

    python scripts/overfit_test.py --data-root <folder> --cache-dir <cache> --out <results folder>

A pipeline sanity check only: train on 8 images for a short *fixed* budget, then score the same 8 images in-sample with
the PQ replica.  If the score stays low the pipeline is investigated; nothing is tuned.

What is exercised end to end: cache -> 512 px arm -> sampler -> dataset -> model -> loss -> optimiser -> whole-image
prediction -> bilinear upsampling -> threshold -> components -> RLE -> PQ against *every* annotator version.

Reference ("oracle") rows: the soft target itself (at the same 512 px resolution) pushed through the same
post-processing.  Images with several annotator versions cannot score PQ 1 against every version, so the oracle is the
honest ceiling for this check.  Settings that are not from the documents are labelled in the printed header.
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from filament import data, losses, model as model_mod, pq_metric, postprocess, train

MEAN, STD = 0.32742, 0.22230  # P1 (Spec S5), recomputed in tests/test_data.py


def score_maps(maps: np.ndarray, names: list[str], ann, gt_rles, thresholds) -> list[dict]:
    rows = []
    version_image = {v: ann.image_of[v] for n in names for v in ann.versions_of[n]}
    gt = {v: gt_rles[v] for v in version_image}
    for t in thresholds:
        cfg = postprocess.PostprocessConfig(threshold=t)
        preds = {n: postprocess.postprocess(m, cfg).rles() for n, m in zip(names, maps)}
        res = pq_metric.score_fast(gt, preds, version_image)
        s = res.summary()
        s["threshold"] = t
        rows.append(s)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=300, help="fixed budget of the sanity check (not from the documents)")
    ap.add_argument("--warmup", type=int, default=30, help="warm-up steps; the documents' 200 would be 2/3 of 300")
    ap.add_argument("--n-images", type=int, default=8)
    ap.add_argument("--selection-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="sampler seed")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4, help="A4.3 value; other values are diagnostics only")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    layout = data.find_data_layout(args.data_root)
    ann = data.load_annotations(layout.train_json)
    cache = data.load_cache(args.cache_dir)
    rng = np.random.default_rng(args.selection_seed)
    pick = sorted(rng.choice(len(cache.train_names), size=args.n_images, replace=False).tolist())
    names = [cache.train_names[i] for i in pick]
    print("device:", device, "| resolution:", args.resolution, "| steps:", args.steps, "| warm-up:", args.warmup,
          "| lr:", args.lr)
    print("images:")
    for i, n in zip(pick, names):
        print("  [%3d] %s  %s  versions=%d" % (i, n, data.instrument_of(n), len(ann.versions_of[n])))

    imgs = np.stack([np.asarray(cache.images_1024[i]) for i in pick])
    tgts = np.stack([np.asarray(cache.targets_1024[i]) for i in pick])
    if args.resolution == 512:
        imgs = np.stack([data.area_downscale2(a) for a in imgs])
        tgts = np.stack([data.area_downscale2(a) for a in tgts])
    ds = data.FilamentDataset(imgs, tgts, MEAN, STD)  # default augmentation: none
    sampler = data.SpecSampler(range(len(pick)), batch_size=len(pick), n_steps=args.steps,
                               resolution=args.resolution, crop_sides=None, seed=args.seed)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)

    net = model_mod.build_model(model_mod.ModelConfig()).to(device)  # resnet34 U-Net, ImageNet weights
    loss_fn = losses.CombinedLoss(losses.LossConfig())
    ocfg = train.OptimConfig(warmup_steps=args.warmup, lr=args.lr)
    t0 = time.time()
    out = train.train_loop(net, loader, loss_fn, ocfg, device, log_every=25)
    print("training: %d steps in %.1f s (%.3f s/step, includes the loader), AMP %s, peak GPU %.2f GiB"
          % (out["steps"], out["seconds"], out["seconds_per_step"], out["amp"], out["peak_gpu_gib"] or 0))

    # in-sample scoring against every annotator version (2048 px, pycocotools rasterisation)
    gt_rles = {v: data.version_rles(ann.version_anns[v]) for n in names for v in ann.versions_of[n]}
    probs = train.predict_probabilities(net, imgs, MEAN, STD, device)
    oracle = tgts.astype(np.float32) / 255.0
    thresholds = (0.3, 0.4, 0.5)
    model_rows = score_maps(probs, names, ann, gt_rles, thresholds)
    oracle_rows = score_maps(oracle, names, ann, gt_rles, thresholds)
    hdr = "%-10s %4s | %6s %6s %6s | %4s %4s %4s | %9s | %10s"
    print(hdr % ("source", "t", "PQ", "SQ", "RQ", "TP", "FP", "FN", "pix.Dice", "pred/image"))
    for lab, rows in (("model", model_rows), ("oracle", oracle_rows)):
        for r in rows:
            print(hdr % (lab, "%.2f" % r["threshold"], "%.3f" % r["PQ"], "%.3f" % r["SQ"], "%.3f" % r["RQ"], r["TP"],
                         r["FP"], r["FN"], "%.3f" % r["pixel_dice_mean"], "%.1f" % r["pred_per_image_mean"]))
    h = out["history"]
    print("loss: first %.4f -> last %.4f (min %.4f); bce %.4f -> %.4f; dice term %.4f -> %.4f"
          % (h["loss"][0], h["loss"][-1], min(h["loss"]), h["bce"][0], h["bce"][-1], h["region"][0], h["region"][-1]))

    result = {
        "images": names, "n_versions": [len(ann.versions_of[n]) for n in names], "steps": args.steps,
        "warmup": args.warmup, "lr": args.lr, "device": device, "train": {k: v for k, v in out.items() if k != "history"},
        "loss_first_last_min": [h["loss"][0], h["loss"][-1], min(h["loss"])],
        "model": model_rows, "oracle": oracle_rows, "total_seconds": time.time() - t0,
    }
    (args.out / "overfit_result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    np.savetxt(args.out / "overfit_loss_curve.csv", np.column_stack([np.arange(1, len(h["loss"]) + 1), h["loss"], h["bce"],
               h["region"], h["lr"]]), delimiter=",", header="step,loss,bce,region,lr", comments="")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(h["loss"], label="total"), ax.plot(h["bce"], label="BCE"), ax.plot(h["region"], label="soft Dice")
        ax.set_xlabel("step"), ax.set_ylabel("loss"), ax.legend(), ax.set_title("8-image overfit, %d px" % args.resolution)
        fig.tight_layout()
        fig.savefig(args.out / "overfit_loss_curve.png", dpi=120)
    except Exception as exc:  # plotting must never fail the check
        print("plot skipped:", exc)


if __name__ == "__main__":
    main()
