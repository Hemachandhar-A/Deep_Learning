"""Inference on the test images: fold models + flips -> averaged probability maps -> filaments -> CSV (Arch section 2).

Steps (decision D10, section 5.2):

1. load the fold models (each model file is checked against its checksum sidecar first);
2. for every test image: run the whole image through every fold model and every flip *of that same image* (S6: only the
   single provided image is used), and average the sigmoid maps;
3. save the averaged maps as float16, so post-processing can be redone without the models;
4. post-process each map (``postprocess.py``, parameters from the config) into disjoint filament masks;
5. write the CSV and run **every** check of ``rle.py``: header, id format, unique ids, decodability, non-empty masks,
   no pixel claimed twice inside an image; plus a round trip of every mask (encode -> decode -> identical).

The CSV is only prepared here.  Uploading to the competition is done by the user (rule 2 of the development prompt).
Disc-based options (``r_max``, ``c_limb``) are refused: the disc recipe is deferred (decision D-f).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import data, io_utils, rle, runinfo
from .config import load_with_local
from .model import ModelConfig, build_model
from .postprocess import PostprocessConfig, postprocess
from .predict import predict_probabilities
from .train import load_model_weights, settings_from_cfg


def load_fold_models(run_dir: Path, folds: list[int], model_cfg: ModelConfig, device: str) -> list[tuple[int, torch.nn.Module, dict]]:
    """Fold models in ``run_dir`` (``fold_<k>/model_last.pt``), each verified against its sidecar before use."""
    out = []
    for k in folds:
        path = Path(run_dir) / ("fold_%d" % k) / "model_last.pt"
        # weights come from the checkpoint, so do not download the ImageNet weights again
        cfg = ModelConfig(**{**vars(model_cfg), "encoder_weights": None})
        model = build_model(cfg).to(device)
        meta = load_model_weights(path, model, device)
        model.eval()
        out.append((k, model, meta))
    return out


def average_over_models(models, images_u8, mean: float, std: float, device: str, amp: bool, flips, batch_size: int = 4, stats: dict | None = None):
    """Mean over fold models (and flips inside each) of the sigmoid maps; float16 ``(N, H, W)``."""
    out = []
    for i in range(0, len(images_u8), batch_size):
        chunk = images_u8[i : i + batch_size]
        acc = None
        for _, model, _ in models:
            p = predict_probabilities(model, chunk, mean, std, device, amp=amp, batch_size=len(chunk), flips=flips,
                                      out_dtype=np.float32, stats=stats)
            acc = p if acc is None else acc + p
        out.append((acc / len(models)).astype(np.float16))
    return np.concatenate(out) if out else np.zeros((0,) + tuple(np.asarray(images_u8).shape[1:]), np.float16)


def maps_to_submission(prob_maps, names: list[str], pp_cfg: PostprocessConfig, csv_path: Path, log=print) -> dict:
    """Post-process every map, write the CSV atomically, and run all checks.  Returns the report."""
    if pp_cfg.needs_disc():
        raise ValueError("r_max / c_limb need the disc recipe, which is deferred (D-f); leave them off")
    rows: list[tuple[str, str]] = []
    comps_per_image: list[int] = []
    areas: list[int] = []
    mismatches = 0
    for name, pm in zip(names, prob_maps):
        res = postprocess(pm, pp_cfg)
        counts = res.counts()
        for k, (c, comp) in enumerate(zip(counts, res.kept), start=1):
            # round trip of every mask: the string must decode to exactly the mask that produced it (inside its
            # bounding box, and with no pixel outside it)
            crop, y0, x0 = res._crop(comp)
            dec = rle.decode_counts(c)
            if not (int(dec.sum()) == int(crop.sum()) and np.array_equal(dec[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]], crop)):
                mismatches += 1
            rows.append((rle.make_filament_ids(name, 1, start=k)[0], c))
            areas.append(comp.area)
        comps_per_image.append(len(counts))
    n = rle.write_submission_csv(csv_path, rows)
    report = rle.check_submission_csv(csv_path, expected_images=names)
    log(report.summary())
    cpi = np.array(comps_per_image) if comps_per_image else np.zeros(1)
    return {
        "csv_rows": n, "images": len(names), "images_with_rows": int((cpi > 0).sum()),
        "csv_check_ok": report.ok, "csv_check_problems": report.problems[:50],
        "round_trip_mismatches": mismatches,
        "components_per_image": {"mean": float(cpi.mean()), "median": float(np.median(cpi)), "max": int(cpi.max()),
                                 "zero": int((cpi == 0).sum())},
        "component_area_px": {"median": float(np.median(areas)) if areas else None,
                              "p5": float(np.percentile(areas, 5)) if areas else None,
                              "p95": float(np.percentile(areas, 95)) if areas else None},
        "reference_gt_filaments_per_version": {"median": 7, "mean": 7.10, "source": "S3"},
    }


def run_inference(cfg: dict, cache: data.Cache, run_dir: Path, out_dir: Path, device: str, folds: list[int] | None = None,
                  flips=("",), threshold: float | None = None, log=print) -> dict:
    """Whole test-set inference from a run folder to ``submission.csv`` + maps + report in ``out_dir``."""
    if cache.test_images_1024 is None:
        raise ValueError("the cache has no test images (build it with the test folder present)")
    s = settings_from_cfg(cfg)
    n_folds = int((cfg.get("folds") or {}).get("n_splits", 5))
    folds = list(range(n_folds)) if folds is None else folds
    pp = dict(cfg.get("postprocess") or {})
    if threshold is not None:
        pp["threshold"] = float(threshold)
    pp_cfg = PostprocessConfig.from_dict(pp)
    t0 = time.perf_counter()
    models = load_fold_models(run_dir, folds, s.model, device)
    log("loaded %d fold models %s" % (len(models), [k for k, _, _ in models]))
    images = cache.test_arm(s.resolution)
    guard: dict = {}
    maps = average_over_models(models, images, s.mean, s.std, device, s.optim.amp, flips, stats=guard)
    t_pred = time.perf_counter() - t0
    io_utils.atomic_save_npy(out_dir / "test_probs.npy", maps)
    io_utils.atomic_write_json(out_dir / "test_probs_index.json", {
        "names": cache.test_names, "resolution": int(images.shape[-1]), "dtype": "float16", "flips": list(flips),
        "fold_models": [{"fold": k, "model_sha256": meta["sha256"], "config_sha256": meta["config_sha256"]}
                        for k, _, meta in models],
    })
    report = maps_to_submission(maps, cache.test_names, pp_cfg, out_dir / "submission.csv", log=log)
    report.update({
        "folds_used": [k for k, _, _ in models], "flips": list(flips), "postprocess": vars(pp_cfg),
        "resolution": int(images.shape[-1]), "predict_seconds": t_pred, "total_seconds": time.perf_counter() - t0,
        "nan_guard": {k: int(guard.get(k, 0)) for k in ("passes", "recomputed", "nonfinite_left")},
        "nonfinite_pixels_in_maps": int(np.count_nonzero(~np.isfinite(maps))),
        "maps_sha256": io_utils.sha256_file(out_dir / "test_probs.npy"),
        "csv_sha256": io_utils.sha256_file(out_dir / "submission.csv"),
        **runinfo.git_info(), **runinfo.gpu_info(device),
        "note": "Prepared for the user to upload; this code never submits (development prompt, rule 2).",
    })
    io_utils.atomic_write_json(out_dir / "infer_report.json", report)
    return report


def plan_inference_after_training(*, dry: bool, folds_requested: list[int] | None, n_folds: int,
                                  results: list[tuple[str, int]], run_dir: Path) -> tuple[bool, list[int], str]:
    """Whether a training session runs inference itself, and with which fold models (Phase 2.2, option A of the Gate 2b
    review, section 5.2).  Returns ``(run, folds, reason)``; the reason is printed either way.

    * **Real session** (``dry`` false): only if every process returned 0, the session was asked to train *all* ``n_folds``
      folds, and every fold has its final model (``model_last.pt``).  Anything less (a failed fold, a session of only some
      folds) means the five-model average that the leaderboard check needs does not exist, so nothing runs.
    * **DRY** (a pipeline check on the PC, ``DRY_`` run ids): any subset of folds is allowed, as long as every requested
      fold returned 0 and has its model, so that the path can be rehearsed on one fold.
    """
    failed = [label for label, code in results if code != 0]
    requested = list(range(n_folds)) if folds_requested is None else sorted(set(int(k) for k in folds_requested))
    with_model = [k for k in requested if (Path(run_dir) / ("fold_%d" % k) / "model_last.pt").exists()]
    if failed:
        return False, [], "not run: %d process(es) failed (%s)" % (len(failed), ", ".join(failed))
    if with_model != requested:
        missing = [k for k in requested if k not in with_model]
        return False, [], "not run: no final model for fold(s) %s" % missing
    if not requested:
        return False, [], "not run: no folds were requested"
    if dry:
        return True, requested, "DRY pipeline check on the folds trained: %s" % requested
    if requested != list(range(n_folds)):
        return False, [], ("not run: this session trained folds %s, inference needs all %d (a partial session only "
                           "trains; run inference when the five models exist)" % (requested, n_folds))
    return True, requested, "all %d processes returned 0 and all %d folds have a final model" % (len(results), n_folds)


def write_infer_manifest(work_dir: Path, files_dir: Path, out: Path) -> Path:
    """``MANIFEST_infer.sha256``: ``<sha256>  <path relative to the work folder>`` for every file the inference wrote
    into ``files_dir`` (maps, CSV, report), for ``sha256sum -c`` after a download."""
    work_dir, files_dir = Path(work_dir), Path(files_dir)
    lines = [(p.relative_to(work_dir).as_posix(), io_utils.sha256_file(p)) for p in sorted(files_dir.rglob("*"))
             if p.is_file() and not p.name.endswith(".tmp")]
    io_utils.atomic_write_text(out, "\n".join("%s  %s" % (h, rel) for rel, h in lines) + "\n")
    return Path(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m filament.infer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--local-config", type=Path, default=None)
    ap.add_argument("--run-dir", required=True, type=Path, help="work/runs/<run_id> with fold_<k>/model_last.pt")
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--flips", nargs="*", default=None, help="flip set, e.g. '' h v hv (default: from config, else none)")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    cfg = load_with_local(a.config, a.local_config)
    flips = tuple(a.flips) if a.flips is not None else tuple((cfg.get("inference") or {}).get("flips", [""]))
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache = data.load_cache(a.cache_dir)
    rep = run_inference(cfg, cache, a.run_dir, a.out_dir, device, a.folds, flips, a.threshold)
    print("infer_report.json: rows %d, images %d, csv_check_ok %s, round-trip mismatches %d"
          % (rep["csv_rows"], rep["images"], rep["csv_check_ok"], rep["round_trip_mismatches"]))
    return 0 if rep["csv_check_ok"] and rep["round_trip_mismatches"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
