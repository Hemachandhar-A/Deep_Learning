"""predict.py (flips), infer.py (fold models -> CSV, all checks) and oof.py (held-out maps -> PQ, grid, bootstrap)."""
import json

import numpy as np
import pytest
import torch

from filament import data, infer, io_utils, oof, predict, rle, train
from filament.postprocess import PostprocessConfig

from synthetic import DRY_INFO, make_cache, make_cfg, rows_for


class Scale(torch.nn.Module):
    def __init__(self, a):
        super().__init__()
        self.a = a

    def forward(self, x):
        return x * self.a


class Skew(torch.nn.Module):
    """Not flip-equivariant: a one-sided horizontal smoothing."""

    def forward(self, x):
        return x + 0.5 * torch.roll(x, 1, dims=-1)


# --------------------------------------------------------------------------------------------- predict.py
def test_flip_average_matches_a_manual_computation_and_identity_is_unchanged():
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, (3, 32, 32), dtype=np.uint8)
    model = Skew()
    base = predict.predict_probabilities(model, imgs, 0.3, 0.2, "cpu", flips=("",), out_dtype=np.float32)
    x = torch.from_numpy(np.stack([data.normalise(a, 0.3, 0.2) for a in imgs]))[:, None]
    assert np.allclose(base, torch.sigmoid(model(x))[:, 0].numpy(), atol=1e-6)
    avg = predict.predict_probabilities(model, imgs, 0.3, 0.2, "cpu", flips=("", "h", "v", "hv"), out_dtype=np.float32)
    manual = 0
    for how in ("", "h", "v", "hv"):
        dims = ([-1] if "h" in how else []) + ([-2] if "v" in how else [])
        xf = torch.flip(x, dims) if dims else x
        p = torch.sigmoid(model(xf))
        manual = manual + (torch.flip(p, dims) if dims else p)
    assert np.allclose(avg, (manual / 4)[:, 0].numpy(), atol=1e-6)
    assert not np.allclose(avg, base, atol=1e-4)  # a non-equivariant model does change with flips
    # an equivariant model (a pointwise function) is unchanged by flips
    same = predict.predict_probabilities(Scale(2.0), imgs, 0.3, 0.2, "cpu", flips=("", "h", "hv"), out_dtype=np.float32)
    ref = predict.predict_probabilities(Scale(2.0), imgs, 0.3, 0.2, "cpu", flips=("",), out_dtype=np.float32)
    assert np.allclose(same, ref, atol=1e-6)
    with pytest.raises(ValueError):
        predict.predict_probabilities(model, imgs, 0.3, 0.2, "cpu", flips=("x",))


def test_average_over_models_is_the_mean_of_probabilities_in_float16():
    imgs = np.random.default_rng(1).integers(0, 256, (5, 32, 32), dtype=np.uint8)
    models = [(0, Scale(1.0), {}), (1, Scale(3.0), {})]
    out = infer.average_over_models(models, imgs, 0.3, 0.2, "cpu", amp=False, flips=("",), batch_size=2)
    x = torch.from_numpy(np.stack([data.normalise(a, 0.3, 0.2) for a in imgs]))
    ref = 0.5 * (torch.sigmoid(x) + torch.sigmoid(3 * x))
    assert out.dtype == np.float16 and out.shape == (5, 32, 32)
    assert np.abs(out.astype(np.float32) - ref.numpy()).max() < 1e-3


# --------------------------------------------------------------------------------------------- infer.py
def _blob_maps():
    m = np.zeros((3, 128, 128), np.float16)
    m[0, 20:40, 10:100] = 0.9  # one filament
    m[0, 70:80, 20:60] = 0.8  # a second one
    m[1, 30:50, 30:50] = 0.95  # one
    # image 2: nothing above the threshold
    m[2] += 0.1
    return m


def test_maps_to_submission_writes_checked_csv_without_rows_for_empty_images(tmp_path):
    names = ["20150101000000Bh", "20150102000000Ch", "20150103000000Lh"]
    rep = infer.maps_to_submission(_blob_maps(), names, PostprocessConfig(threshold=0.5), tmp_path / "s.csv",
                                   log=lambda *_: None)
    assert rep["csv_rows"] == 3 and rep["csv_check_ok"] and rep["round_trip_mismatches"] == 0
    assert rep["images"] == 3 and rep["images_with_rows"] == 2 and rep["components_per_image"]["zero"] == 1
    header, rows = rle.read_submission_csv(tmp_path / "s.csv")
    assert header == ["filament_id", "segmentation_rle"]
    assert [r[0] for r in rows] == [names[0] + "_1", names[0] + "_2", names[1] + "_1"]
    assert rle.check_submission_csv(tmp_path / "s.csv", expected_images=names).ok
    # a CSV for an image that is not a test image would be caught by the checker
    assert not rle.check_submission_csv(tmp_path / "s.csv", expected_images=names[1:]).ok


def test_disc_based_settings_are_refused(tmp_path):
    with pytest.raises(ValueError, match="deferred"):
        infer.maps_to_submission(_blob_maps(), ["a", "b", "c"], PostprocessConfig(c_limb=0.9), tmp_path / "s.csv",
                                 log=lambda *_: None)


def test_run_inference_from_trained_fold_models(tmp_path):
    cache = make_cache(n=12)
    rng = np.random.default_rng(3)
    test_imgs = rng.integers(20, 200, (4, 64, 64), dtype=np.uint8)
    cache.test_images_1024 = test_imgs
    cache.test_names = ["2016%010d%s" % (i, ("Bh", "Ch")[i % 2]) for i in range(4)]
    cfg = make_cfg(train={"steps": 3, "checkpoint_every": 0}, postprocess={"threshold": 0.3})
    cfg["folds"]["n_splits"] = 2
    m = train.run_training(cfg, cache, rows_for(cache), DRY_INFO, [0, 1], tmp_path / "w", tmp_path / "r", "cpu",
                           dry_run=True, log=lambda *_: None)
    run_dir = tmp_path / "w" / "runs" / m["run_id"]
    rep = infer.run_inference(cfg, cache, run_dir, tmp_path / "out", "cpu", flips=("", "h"), log=lambda *_: None)
    assert rep["csv_check_ok"] and rep["round_trip_mismatches"] == 0 and rep["images"] == 4
    assert rep["folds_used"] == [0, 1] and rep["flips"] == ["", "h"]
    idx = json.loads((tmp_path / "out" / "test_probs_index.json").read_text(encoding="utf-8"))
    assert idx["names"] == cache.test_names and len(idx["fold_models"]) == 2
    assert idx["fold_models"][0]["model_sha256"] == m["folds"]["0"]["model_sha256"]
    maps = np.load(tmp_path / "out" / "test_probs.npy")
    assert maps.dtype == np.float16 and maps.shape == (4, 32, 32)
    assert rep["csv_sha256"] == io_utils.sha256_file(tmp_path / "out" / "submission.csv")
    # a damaged model file is refused, not silently used
    mp = run_dir / "fold_1" / "model_last.pt"
    mp.write_bytes(b"\x00" * mp.stat().st_size)
    with pytest.raises(IOError, match="not valid"):
        infer.run_inference(cfg, cache, run_dir, tmp_path / "out2", "cpu", log=lambda *_: None)


# --------------------------------------------------------------------------------------------- oof.py
def _synthetic_annotations():
    names = ["20150101000000Bh", "20150102000000Ch", "20150103000000Lh"]
    rects = {  # image -> list of versions, each a list of (x0, y0, x1, y1) at 2048 px
        names[0]: [[(100, 200, 500, 300), (800, 900, 1200, 1100)], [(110, 205, 500, 300)]],
        names[1]: [[(600, 600, 1000, 700)]],
        names[2]: [[(300, 1200, 700, 1300), (1400, 400, 1800, 500)], [(300, 1200, 700, 1300), (1400, 400, 1800, 500)]],
    }
    versions_of, image_of, version_anns = {}, {}, {}
    for n, vs in rects.items():
        versions_of[n] = []
        for j, v in enumerate(vs):
            vid = "%02d-%s" % (j + 1, n)
            versions_of[n].append(vid)
            image_of[vid] = n
            version_anns[vid] = [{"segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]], "area": (x1 - x0) * (y1 - y0)}
                                 for x0, y0, x1, y1 in v]
    ann = data.Annotations(image_names=names, versions_of=versions_of, image_of=image_of, version_anns=version_anns)
    return ann, rects


def _perfect_maps(ann, rects, shape=512):
    f = 2048 // shape
    maps = []
    for n in ann.image_names:
        m = np.zeros((shape, shape), np.float16)
        for v in rects[n]:  # union of the versions' rectangles, like a soft target thresholded at its support
            for x0, y0, x1, y1 in v:
                m[y0 // f : y1 // f, x0 // f : x1 // f] = 1.0
        maps.append(m)
    return np.stack(maps)


def test_load_oof_orders_and_validates(tmp_path):
    for k, nm in enumerate((["b", "a"], ["c"])):
        fd = tmp_path / ("fold_%d" % k)
        fd.mkdir()
        io_utils.atomic_save_npy(fd / "oof_probs.npy", np.full((len(nm), 8, 8), k, np.float16))
        io_utils.atomic_write_json(fd / "oof_index.json", {"names": nm})
    names, maps = oof.load_oof(tmp_path)
    assert names == ["b", "a", "c"] and maps.shape == (3, 8, 8) and float(maps[2, 0, 0]) == 1.0
    assert oof.load_oof(tmp_path, folds=[1])[0] == ["c"]
    with pytest.raises(FileNotFoundError):
        oof.load_oof(tmp_path / "nothing")


def test_grid_is_the_cartesian_product_and_rejects_unknown_keys():
    base = PostprocessConfig()
    s = oof.grid_settings(base, {"threshold": [0.3, 0.4, 0.5], "a_min": [0, 300], "hysteresis": [None, [0.3, 0.6]]})
    assert len(s) == 12 and {x.threshold for x in s} == {0.3, 0.4, 0.5}
    assert any(x.hysteresis == (0.3, 0.6) for x in s)
    assert oof.grid_settings(base, None) == [base]
    with pytest.raises(ValueError):
        oof.grid_settings(base, {"threshold": [0.5], "nonsense": [1]})


def test_scoring_reproduces_a_perfect_and_an_empty_baseline_and_is_deterministic():
    ann, rects = _synthetic_annotations()
    maps = _perfect_maps(ann, rects)
    settings = oof.grid_settings(PostprocessConfig(), {"threshold": [0.3, 0.5]})
    res = oof.score_settings(maps, ann.image_names, ann, settings, n_boot=50, seed=1, log=None)
    assert len(res) == 2
    for r in res:
        assert r["pq"] > 0.8 and r["FN"] == 0 and r["TP"] >= 6  # 2+1+1+2+2 filaments over 5 versions: all matched
        assert r["pq_ci_lo"] <= r["pq"] <= r["pq_ci_hi"]
        assert r["params"]["threshold"] in (0.3, 0.5)
    again = oof.score_settings(maps, ann.image_names, ann, settings, n_boot=50, seed=1, log=None)
    assert [x["pq"] for x in again] == [x["pq"] for x in res]
    assert [x["pq_ci_lo"] for x in again] == [x["pq_ci_lo"] for x in res]
    empty = oof.score_settings(np.zeros_like(maps), ann.image_names, ann, settings[:1], n_boot=10, log=None)[0]
    assert empty["pq"] == 0.0 and empty["TP"] == 0 and empty["FN"] == 8 and empty["FP"] == 0
    assert oof.best_setting(res)["pq"] == max(r["pq"] for r in res)
    assert "result" not in oof.jsonable(res)[0] and json.dumps(oof.jsonable(res))


def test_scoring_refuses_disc_settings_unless_allowed():
    ann, rects = _synthetic_annotations()
    maps = _perfect_maps(ann, rects)
    with pytest.raises(ValueError, match="deferred"):
        oof.score_settings(maps, ann.image_names, ann, [PostprocessConfig(r_max=0.97)], log=None)


def test_score_run_gives_the_small_file_the_reviewer_decides_on(tmp_path):
    """Phase 2.1 R3a: held-out maps of the folds present -> one JSON-able dict at t 0.30, 0.40, 0.50 (and 0.5 default)."""
    ann, rects = _synthetic_annotations()
    maps = _perfect_maps(ann, rects)
    names = ann.image_names
    half = len(names) // 2
    for k, sl in enumerate((slice(0, half), slice(half, None))):
        fd = tmp_path / "run" / ("fold_%d" % k)
        fd.mkdir(parents=True)
        io_utils.atomic_save_npy(fd / "oof_probs.npy", maps[sl])
        io_utils.atomic_write_json(fd / "oof_index.json", {"names": names[sl]})
    out = oof.score_run(tmp_path / "run", ann, n_boot=20, log=None)
    assert out["run"] == "run" and out["folds"] == [0, 1] and out["n_images"] == len(names) and out["n_boot"] == 20
    assert [r["params"]["threshold"] for r in out["results"]] == [0.3, 0.4, 0.5]  # the default is among them already
    assert all(r["pq"] > 0.8 for r in out["results"]) and out["best"]["pq"] == max(r["pq"] for r in out["results"])
    assert len(json.dumps(out)) < 20000  # a few kilobytes, not megabytes of maps
    only0 = oof.score_run(tmp_path / "run", ann, thresholds=(0.5,), n_boot=10, folds=[0], log=None)
    assert only0["folds"] == [0] and only0["n_images"] == half and len(only0["results"]) == 1
    with pytest.raises(FileNotFoundError):
        oof.score_run(tmp_path / "nothing", ann, n_boot=5, log=None)


def _run_dir_with_models(tmp_path, folds):
    for k in folds:
        fd = tmp_path / "run" / ("fold_%d" % k)
        fd.mkdir(parents=True)
        (fd / "model_last.pt").write_bytes(b"x")
    return tmp_path / "run"


def test_inference_after_training_runs_only_with_all_five_models_in_a_real_session(tmp_path):
    """Phase 2.2, option A (Gate 2b review 5.2): the rule for when a training session runs the inference itself."""
    plan = infer.plan_inference_after_training
    ok = [("e1 fold %d" % k, 0) for k in range(5)]
    run = _run_dir_with_models(tmp_path, range(5))
    go, folds_, why = plan(dry=False, folds_requested=None, n_folds=5, results=ok, run_dir=run)
    assert go and folds_ == [0, 1, 2, 3, 4] and "all 5" in why
    assert plan(dry=False, folds_requested=[0, 1, 2, 3, 4], n_folds=5, results=ok, run_dir=run)[0]
    # one process failed: no inference, and the reason names it
    bad = ok[:2] + [("e1 fold 2", 1)] + ok[3:]
    go, _, why = plan(dry=False, folds_requested=None, n_folds=5, results=bad, run_dir=run)
    assert not go and "failed" in why and "e1 fold 2" in why
    # a session of only some folds trains but does not infer
    part = _run_dir_with_models(tmp_path / "part", [0, 1])
    go, _, why = plan(dry=False, folds_requested=[0, 1], n_folds=5, results=ok[:2], run_dir=part)
    assert not go and "needs all 5" in why
    # all five asked and returned 0 but a model file is missing
    missing = _run_dir_with_models(tmp_path / "missing", [0, 1, 2, 3])
    go, _, why = plan(dry=False, folds_requested=None, n_folds=5, results=ok, run_dir=missing)
    assert not go and "no final model for fold(s) [4]" in why
    # DRY (a pipeline check on the PC) may use any subset that trained cleanly
    one = _run_dir_with_models(tmp_path / "one", [0])
    go, folds_, why = plan(dry=True, folds_requested=[0], n_folds=5, results=ok[:1], run_dir=one)
    assert go and folds_ == [0] and "DRY" in why
    assert not plan(dry=True, folds_requested=[0, 9], n_folds=5, results=ok[:1] + [("e1 fold 9", 1)], run_dir=one)[0]
    assert not plan(dry=True, folds_requested=[], n_folds=5, results=[], run_dir=one)[0]


def test_infer_manifest_lists_every_new_file_with_its_checksum(tmp_path):
    work = tmp_path / "w"
    out = work / "submission_candidate"
    out.mkdir(parents=True)
    (out / "submission.csv").write_text("id,rle\n", encoding="utf-8")
    io_utils.atomic_save_npy(out / "test_probs.npy", np.zeros((2, 4, 4), np.float16))
    mf = infer.write_infer_manifest(work, out, work / "results" / "R" / "MANIFEST_infer.sha256")
    lines = mf.read_text(encoding="utf-8").splitlines()
    assert [l.split("  ", 1)[1] for l in lines] == ["submission_candidate/submission.csv", "submission_candidate/test_probs.npy"]
    for l in lines:
        h, rel = l.split("  ", 1)
        assert io_utils.sha256_file(work / rel) == h
    assert mf.read_bytes().endswith(b"\n") and b"\r" not in mf.read_bytes()
