"""train.py (Gate 1b review B1-B3, Follow-up review P3): folds verification, metrics fields, atomic hash-checked
checkpoints, exact resume, completed folds are not retrained.  CPU, tiny synthetic cache, random-weight U-Net."""
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from filament import data, folds, io_utils, runinfo, train
from synthetic import DRY_INFO, make_cache, make_cfg, rows_for


# ----------------------------------------------------------------------------------------- B1: folds verification
def _folds_file(tmp_path):
    v = {"2015%010d%s" % (i, ("Bh", "Ch", "Lh", "Mh", "Th", "Uh")[i % 6]): [0] * (1 + i % 3) for i in range(300)}
    rows = folds.make_folds(v)
    p = tmp_path / "folds.csv"
    folds.write_folds_csv(p, rows)
    io_utils.atomic_write_json(folds.sidecar_path(p), {**folds.provenance(p), "strategy": "balanced", "seed": 42})
    return p


def test_verify_for_run_accepts_the_right_hash_and_reports_provenance(tmp_path):
    p = _folds_file(tmp_path)
    info = folds.verify_for_run(p)  # expected hash from the sidecar
    assert info["folds_verified"] and info["expected_sha256_source"] == "sidecar"
    assert info["folds_csv_sha256"] == io_utils.sha256_file(p)
    assert info["folds_strategy"] == "balanced" and info["folds_seed"] == 42 and info["scikit_learn_version"]
    assert folds.verify_for_run(p, configured_sha256=info["folds_csv_sha256"])["expected_sha256_source"] == "config"


def test_wrong_hash_is_refused_even_with_the_unverified_flag(tmp_path):
    p = _folds_file(tmp_path)
    with pytest.raises(folds.FoldsVerificationError):  # the config expects another hash than the sidecar records
        folds.verify_for_run(p, configured_sha256="0" * 64)
    side = folds.sidecar_path(p).read_text(encoding="utf-8")
    folds.sidecar_path(p).unlink()  # only the config's expectation is left: the file does not match it
    with pytest.raises(folds.FoldsVerificationError, match="not the split"):
        folds.verify_for_run(p, configured_sha256="0" * 64)
    with pytest.raises(folds.FoldsVerificationError, match="not the split"):
        folds.verify_for_run(p, configured_sha256="0" * 64, allow_unverified=True)
    folds.sidecar_path(p).write_text(side, encoding="utf-8")
    # a damaged file (one fold number changed) no longer matches the sidecar
    text = p.read_text(encoding="utf-8")
    p.write_text(text.replace(",0\n", ",1\n", 1), encoding="utf-8", newline="")
    with pytest.raises(folds.FoldsVerificationError):
        folds.verify_for_run(p)
    # a zero-filled file (the damage seen on 2026-10-12)
    p.write_bytes(b"\x00" * 100)
    with pytest.raises(folds.FoldsVerificationError):
        folds.verify_for_run(p)


def test_no_expected_hash_is_refused_unless_flagged_and_two_expectations_must_agree(tmp_path):
    p = _folds_file(tmp_path)
    folds.sidecar_path(p).unlink()
    with pytest.raises(folds.FoldsVerificationError, match="no expected SHA-256"):
        folds.verify_for_run(p)
    info = folds.verify_for_run(p, allow_unverified=True)
    assert info["folds_verified"] is False and info["folds_csv_sha256"]
    io_utils.atomic_write_json(folds.sidecar_path(p), {"folds_csv_sha256": "a" * 64})
    with pytest.raises(folds.FoldsVerificationError, match="disagree"):
        folds.verify_for_run(p, configured_sha256="b" * 64)
    with pytest.raises(FileNotFoundError):
        folds.verify_for_run(tmp_path / "missing.csv")


def test_fields_of_metrics_json_and_no_machine_paths(tmp_path):
    cache = make_cache()
    p = _folds_file(tmp_path)
    info = folds.verify_for_run(p)
    cfg = make_cfg()
    m = train.run_training(cfg, cache, rows_for(cache), info, [0], tmp_path / "work", tmp_path / "results", "cpu",
                           dry_run=True, log=lambda *_: None)
    for key in ("folds_csv_sha256", "scikit_learn_version", "folds_strategy", "folds_seed", "git_commit", "lr",
                "training_seed", "gpu_name", "seconds_per_step", "total_time_seconds", "run_id", "environment",
                "steps", "batch_size", "resolution", "folds"):
        assert key in m, key
    assert m["folds_csv_sha256"] == info["folds_csv_sha256"] and m["lr"] == 1e-3 and m["training_seed"] == 1
    assert m["gpu_name"] == "cpu" and m["seconds_per_step"] > 0 and m["total_time_seconds"] > 0
    on_disk = json.loads((tmp_path / "results" / m["run_id"] / "metrics.json").read_text(encoding="utf-8"))
    assert on_disk["folds_csv_sha256"] == info["folds_csv_sha256"]
    # review Q13: no absolute path, user name or file name in what may be committed
    for f in (tmp_path / "results" / m["run_id"]).iterdir():
        text = f.read_text(encoding="utf-8")
        assert str(tmp_path) not in text and "secret" not in text and "should/not" not in text, f.name
        assert not re.search(r"20\d{12}[A-Z][a-z]", text), "an image name leaked into %s" % f.name


# ----------------------------------------------------------------------------------------- B2: --lr in id and metrics
def test_lr_override_is_part_of_the_run_id_and_recorded(tmp_path):
    cache = make_cache()
    cfg = make_cfg()
    m = train.run_training(cfg, cache, rows_for(cache), DRY_INFO, [0], tmp_path / "w", tmp_path / "r", "cpu",
                           lr=3e-4, steps=2, dry_run=True, log=lambda *_: None)
    assert "lr3e-4" in m["run_id"] and m["lr"] == 3e-4 and m["run_id"].startswith("DRY_T_r512_")
    saved = (tmp_path / "r" / m["run_id"] / "config.yaml").read_text(encoding="utf-8")
    assert "3e-4" in saved or "0.0003" in saved
    assert runinfo.run_id_lr(1e-4) == "1e-4" and runinfo.run_id_lr(3e-5) == "3e-5" and runinfo.run_id_lr(1e-3) == "1e-3"
    assert runinfo.make_run_id("E1", 512, 1e-4, 2) == "E1_r512_lr1e-4_seed2"


def test_training_seed_changes_the_trained_weights_only_through_the_seed(tmp_path):
    cache = make_cache()
    cfg = make_cfg(train={"steps": 4})
    a = train.run_training(cfg, cache, rows_for(cache), DRY_INFO, [0], tmp_path / "w", tmp_path / "r", "cpu", seed=1,
                           dry_run=True, log=lambda *_: None)
    b = train.run_training(cfg, cache, rows_for(cache), DRY_INFO, [0], tmp_path / "w", tmp_path / "r", "cpu", seed=2,
                           dry_run=True, log=lambda *_: None)
    assert a["run_id"] != b["run_id"] and "seed1" in a["run_id"] and "seed2" in b["run_id"]
    assert a["folds"]["0"]["model_sha256"] != b["folds"]["0"]["model_sha256"]


# ----------------------------------------------------------------------------------------- outputs of a fold
def test_fold_outputs(tmp_path):
    cache = make_cache()
    rows = rows_for(cache)
    m = train.run_training(make_cfg(), cache, rows, DRY_INFO, [0, 1], tmp_path / "w", tmp_path / "r", "cpu",
                           dry_run=True, log=lambda *_: None)
    assert m["folds_trained"] == [0, 1]
    for k in (0, 1):
        fd = tmp_path / "w" / "runs" / m["run_id"] / ("fold_%d" % k)
        oof = np.load(fd / "oof_probs.npy")
        idx = json.loads((fd / "oof_index.json").read_text(encoding="utf-8"))
        n_val = sum(r["fold"] == k for r in rows)
        assert oof.dtype == np.float16 and oof.shape == (n_val, 32, 32) and 0.0 <= oof.min() <= oof.max() <= 1.0
        assert len(idx["names"]) == n_val and all(rows[[r["image"] for r in rows].index(n)]["fold"] == k for n in idx["names"])
        ok, why, meta = train.check_checkpoint(fd / "model_last.pt")
        assert ok and meta["kind"] == "final" and meta["step"] == 6
        assert not list(fd.glob("ckpt_step*.pt"))  # resume files are removed once the fold is complete
        sd = torch.load(fd / "model_last.pt", weights_only=True)["model"]
        assert any(v.dtype == torch.float32 for v in sd.values())


def test_steps_not_set_is_refused(tmp_path):
    cache = make_cache()
    cfg = make_cfg()
    cfg["train"]["steps"] = None
    with pytest.raises(ValueError, match="steps is not set"):
        train.run_training(cfg, cache, rows_for(cache), DRY_INFO, [0], tmp_path / "w", tmp_path / "r", "cpu",
                           dry_run=True, log=lambda *_: None)


def test_a_complete_fold_is_not_trained_again(tmp_path):
    cache = make_cache()
    kw = dict(work_dir=tmp_path / "w", results_dir=tmp_path / "r", device="cpu", dry_run=True)
    m1 = train.run_training(make_cfg(), cache, rows_for(cache), DRY_INFO, [0], log=lambda *_: None, **kw)
    fd = tmp_path / "w" / "runs" / m1["run_id"] / "fold_0"
    stamp = (fd / "model_last.pt").stat().st_mtime_ns
    msgs = []
    m2 = train.run_training(make_cfg(), cache, rows_for(cache), DRY_INFO, [0], log=msgs.append, **kw)
    assert any("already complete" in str(x) for x in msgs)
    assert (fd / "model_last.pt").stat().st_mtime_ns == stamp
    assert m2["folds"]["0"]["model_sha256"] == m1["folds"]["0"]["model_sha256"]


# ----------------------------------------------------------------------------------------- B3: checkpoints and resume
class _Reboot(RuntimeError):
    pass


def test_resume_after_a_simulated_reboot_equals_an_uninterrupted_run(tmp_path, monkeypatch):
    cache = make_cache()
    rows = rows_for(cache)
    cfg = make_cfg(train={"steps": 6, "checkpoint_every": 2})
    kw = dict(device="cpu", dry_run=True, log=lambda *_: None)
    ref = train.run_training(cfg, cache, rows, DRY_INFO, [0], tmp_path / "w_ref", tmp_path / "r_ref", **kw)

    real_save = train.save_resume_checkpoint

    def crash_after_step_4(fdir, step, *a, **k):
        p = real_save(fdir, step, *a, **k)
        if step == 4:
            raise _Reboot("power lost right after the step-4 checkpoint")
        return p

    monkeypatch.setattr(train, "save_resume_checkpoint", crash_after_step_4)
    with pytest.raises(_Reboot):
        train.run_training(cfg, cache, rows, DRY_INFO, [0], tmp_path / "w", tmp_path / "r", **kw)
    fd = tmp_path / "w" / "runs" / ref["run_id"] / "fold_0"
    assert sorted(p.name for p in fd.glob("ckpt_step*.pt")) == ["ckpt_step000002.pt", "ckpt_step000004.pt"]
    assert not (fd / "model_last.pt").exists()

    monkeypatch.setattr(train, "save_resume_checkpoint", real_save)
    res = train.run_training(cfg, cache, rows, DRY_INFO, [0], tmp_path / "w", tmp_path / "r", **kw)
    assert res["folds"]["0"]["resumed_from_step"] == 4 and res["folds"]["0"]["steps_run"] == 2
    a = torch.load(tmp_path / "w_ref" / "runs" / ref["run_id"] / "fold_0" / "model_last.pt", weights_only=True)["model"]
    b = torch.load(fd / "model_last.pt", weights_only=True)["model"]
    assert a.keys() == b.keys()
    worst = max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
    assert worst < 1e-5, worst  # same batches, same schedule: the resumed run ends where the uninterrupted one ends
    h_ref = json.loads((tmp_path / "w_ref" / "runs" / ref["run_id"] / "fold_0" / "loss_history.json").read_text())
    h_res = json.loads((fd / "loss_history.json").read_text())
    assert len(h_res["loss"]) == 6 and np.allclose(h_ref["loss"], h_res["loss"], atol=1e-5)


def test_damaged_checkpoint_is_skipped_and_another_configuration_is_refused(tmp_path):
    cache = make_cache()
    cfg = make_cfg()
    s = train.settings_from_cfg(cfg)
    sha = s.identity_sha256(0, None)
    model = train.build_model(s.model)
    opt, sched = train.make_optimizer(model, s.optim, 6)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    fdir = tmp_path / "fold_0"
    fdir.mkdir()
    hist = {"loss": [1.0], "bce": [1.0], "region": [1.0], "lr": [1e-3]}
    for step in (2, 4):
        train.save_resume_checkpoint(fdir, step, model, opt, sched, scaler, hist, sha, keep=5)
    p2, p4 = fdir / "ckpt_step000002.pt", fdir / "ckpt_step000004.pt"
    assert train.check_checkpoint(p4, sha)[0]
    p4.write_bytes(b"\x00" * p4.stat().st_size)  # zero-filled, same size: the K15 damage
    ok, why, _ = train.check_checkpoint(p4)
    assert not ok and "SHA-256" in why
    msgs = []
    path, meta = train.find_resume_checkpoint(fdir, sha, msgs.append)
    assert path == p2 and meta["step"] == 2 and any("ckpt_step000004" in m for m in msgs)
    with pytest.raises(train.CheckpointConfigMismatch):
        train.find_resume_checkpoint(fdir, "f" * 64)
    p2.write_bytes(p2.read_bytes()[:100])  # truncated
    assert train.find_resume_checkpoint(fdir, sha, lambda *_: None) is None
    # only the two newest are kept
    for step in (6, 8, 10):
        train.save_resume_checkpoint(fdir, step, model, opt, sched, scaler, hist, sha, keep=2)
    assert sorted(p.name for p in fdir.glob("ckpt_step*.pt")) == ["ckpt_step000008.pt", "ckpt_step000010.pt"]


def test_load_model_weights_checks_the_sidecar(tmp_path):
    cache = make_cache()
    m = train.run_training(make_cfg(), cache, rows_for(cache), DRY_INFO, [0], tmp_path / "w", tmp_path / "r", "cpu",
                           dry_run=True, log=lambda *_: None)
    p = tmp_path / "w" / "runs" / m["run_id"] / "fold_0" / "model_last.pt"
    model = train.build_model(train.settings_from_cfg(make_cfg()).model)
    assert train.load_model_weights(p, model, "cpu")["kind"] == "final"
    p.write_bytes(b"\x00" * p.stat().st_size)
    with pytest.raises(IOError, match="not valid"):
        train.load_model_weights(p, model, "cpu")


# ----------------------------------------------------------------------------------------- speed measurement, CLI
def test_measure_speed_reports_the_derivation_and_applies_nothing(tmp_path):
    cache = make_cache()
    r = train.measure_speed(make_cfg(), cache, rows_for(cache), 0, "cpu", n_steps=4, warmup=1, log=lambda *_: None)
    assert r["measured_steps"] == 4 and r["seconds_per_step_mean"] > 0
    assert r["derived_steps_for_24_min_per_fold"] == int(1440.0 / r["seconds_per_step_mean"])
    assert "not applied" in r["derived_note"]


def test_command_line_dry_run_end_to_end(tmp_path, capsys):
    cache = make_cache()
    cdir = tmp_path / "cache"
    cdir.mkdir()
    np.save(cdir / "train_images_1024.npy", cache.images_1024)
    np.save(cdir / "train_targets_1024.npy", cache.targets_1024)
    (cdir / "cache_index.json").write_text(json.dumps({"train_names": cache.train_names, "test_names": [],
                                                       "n_versions": cache.n_versions}), encoding="utf-8")
    cfgp = tmp_path / "cfg.yaml"
    import yaml

    cfgp.write_text(yaml.safe_dump(make_cfg()), encoding="utf-8")
    rc = train.main(["--config", str(cfgp), "--cache-dir", str(cdir), "--work-dir", str(tmp_path / "w"),
                     "--results-dir", str(tmp_path / "r"), "--dry-run-folds", "2", "--folds", "1", "--steps", "3",
                     "--device", "cpu"])
    assert rc == 0
    runs = list((tmp_path / "r").iterdir())
    assert len(runs) == 1 and runs[0].name.startswith("DRY_")
    assert json.loads((runs[0] / "metrics.json").read_text(encoding="utf-8"))["folds_trained"] == [1]
    # without --dry-run-folds a missing folds file is an error, not a silent default
    with pytest.raises((SystemExit, FileNotFoundError)):  # the configured folds file does not exist
        train.main(["--config", str(cfgp), "--cache-dir", str(cdir), "--work-dir", str(tmp_path / "w2"),
                    "--results-dir", str(tmp_path / "r2"), "--steps", "3", "--device", "cpu"])


def test_publishable_config_strips_paths():
    c = train.publishable_config(make_cfg())
    assert "paths" not in c and "file" not in c["folds"] and c["folds"]["seed"] == 42


# ----------------------------------------------------------------------------------------- Phase 2.1
def test_logged_learning_rate_is_the_rate_of_the_step_that_was_taken(tmp_path):
    """Gate 2 finding G2-6: the log and loss_history.json used to show the rate of the NEXT step."""
    cache = make_cache()
    cfg = make_cfg(train={"steps": 6})
    m = train.run_training(cfg, cache, rows_for(cache), DRY_INFO, [0], tmp_path / "w", tmp_path / "r", "cpu",
                           dry_run=True, log=lambda *_: None)
    hist = json.loads((tmp_path / "w" / "runs" / m["run_id"] / "fold_0" / "loss_history.json").read_text(encoding="utf-8"))
    expected = [1e-3 * train.lr_factor(s, 6, 2) for s in range(6)]  # lr 1e-3, warm-up 2 steps (make_cfg)
    assert hist["lr"] == pytest.approx(expected)
    assert hist["lr"][0] == pytest.approx(0.5e-3)  # the first step ran at half the base rate, not at the next step's rate


def test_fold_manifest_and_run_manifest_checksum_every_output_file(tmp_path):
    """Phase 2.1 R3b: each piece can be fetched alone and checked."""
    cache = make_cache()
    work, results = tmp_path / "w", tmp_path / "w" / "results"
    m = train.run_training(make_cfg(), cache, rows_for(cache), DRY_INFO, [0, 1], work, results, "cpu", dry_run=True,
                           log=lambda *_: None)
    rid = m["run_id"]
    for k in (0, 1):
        fd = work / "runs" / rid / ("fold_%d" % k)
        fm = json.loads((fd / "fold_manifest.json").read_text(encoding="utf-8"))
        assert set(fm["files"]) == set(train.FOLD_FILES) and fm["fold_dir"] == "fold_%d" % k
        for name, e in fm["files"].items():
            assert e["sha256"] == io_utils.sha256_file(fd / name) and e["bytes"] == (fd / name).stat().st_size
    mf = train.write_run_manifest(work, rid)
    assert mf == results / rid / "MANIFEST.sha256" and mf.read_bytes().endswith(b"\n") and b"\r" not in mf.read_bytes()
    entries = {l.split("  ", 1)[1]: l.split("  ", 1)[0] for l in mf.read_text(encoding="utf-8").splitlines()}
    assert "runs/%s/fold_0/oof_probs.npy" % rid in entries and "results/%s/metrics.json" % rid in entries
    assert "results/%s/MANIFEST.sha256" % rid not in entries and not [p for p in entries if "ckpt_step" in p]
    for rel, h in entries.items():
        assert io_utils.sha256_file(work / rel) == h, rel
    assert list(entries) == sorted(entries)  # stable order


def test_aggregation_rebuilds_the_fold_keys_after_a_lost_update(tmp_path):
    """Gate 2 finding G2-4: two fold processes ending together can leave a metrics.json that lacks one fold."""
    cache = make_cache()
    work, results = tmp_path / "w", tmp_path / "r"
    m = train.run_training(make_cfg(), cache, rows_for(cache), DRY_INFO, [0, 1], work, results, "cpu", dry_run=True,
                           log=lambda *_: None)
    mp = results / m["run_id"] / "metrics.json"
    damaged = json.loads(mp.read_text(encoding="utf-8"))
    damaged["folds_trained"] = [0]
    damaged["folds"].pop("1")
    damaged["total_time_seconds"] = 0.0
    mp.write_text(json.dumps(damaged), encoding="utf-8")
    agg = train.aggregate_run_metrics(work / "runs" / m["run_id"], results)
    assert agg["folds_trained"] == [0, 1] and set(agg["folds"]) == {"0", "1"} and agg["total_time_seconds"] > 0
    for k in ("run_id", "lr", "folds_csv_sha256", "git_commit", "steps", "environment"):
        assert agg[k] == m[k], k  # nothing else changes
    assert json.loads(mp.read_text(encoding="utf-8"))["folds_trained"] == [0, 1]
    with pytest.raises(FileNotFoundError, match="no fold of this run has finished"):
        train.aggregate_run_metrics(work / "runs" / "NO_SUCH_RUN", results)


def test_run_record_has_budget_override_flag_hardware_and_cache_location(tmp_path):
    """Phase 2.1: step budget, whether the run differs from it, CPU/GPU count and names, cache location, libraries."""
    cache = make_cache()
    cfg = make_cfg(train={"steps": 6})
    same = train.run_training(cfg, cache, rows_for(cache), DRY_INFO, [0], tmp_path / "w", tmp_path / "r", "cpu",
                              dry_run=True, extras={"cache_location": runinfo.cache_location_label("/kaggle/temp/cache")},
                              log=lambda *_: None)
    assert same["step_budget"] == 6 and same["steps_differs_from_budget"] is False and same["steps"] == 6
    assert same["cache_location"] == "/kaggle/temp/cache"
    other = train.run_training(cfg, cache, rows_for(cache), DRY_INFO, [0], tmp_path / "w2", tmp_path / "r2", "cpu",
                               steps=2, dry_run=True, log=lambda *_: None)
    assert other["step_budget"] == 6 and other["steps_differs_from_budget"] is True and other["steps"] == 2
    env = same["environment"]
    assert isinstance(env["cpu_count"], int) and env["cpu_count"] >= 1
    assert env["gpu_count"] == len(env["gpu_names"]) >= 0
    for lib in ("torch", "numpy", "pillow", "segmentation_models_pytorch", "timm", "scikit-learn", "huggingface_hub"):
        assert env["packages"].get(lib), lib
    assert any(env["packages"].get(k) for k in ("opencv-python-headless", "opencv-python"))


def test_command_line_mark_dry_keeps_the_folds_verified_and_prefixes_the_run_id(tmp_path):
    import yaml

    cache = make_cache()
    cdir = tmp_path / "cache"
    cdir.mkdir()
    np.save(cdir / "train_images_1024.npy", cache.images_1024)
    np.save(cdir / "train_targets_1024.npy", cache.targets_1024)
    (cdir / "cache_index.json").write_text(json.dumps({"train_names": cache.train_names, "test_names": [],
                                                       "n_versions": cache.n_versions}), encoding="utf-8")
    fp = tmp_path / "folds.csv"
    folds.write_folds_csv(fp, folds.make_folds({n: [0] * v for n, v in zip(cache.train_names, cache.n_versions)}, 2))
    io_utils.atomic_write_json(folds.sidecar_path(fp), {**folds.provenance(fp), "strategy": "balanced", "seed": 42})
    cfgp = tmp_path / "cfg.yaml"
    cfgp.write_text(yaml.safe_dump(make_cfg()), encoding="utf-8")
    base = ["--config", str(cfgp), "--cache-dir", str(cdir), "--folds-csv", str(fp), "--fold", "0", "--steps", "3",
            "--device", "cpu"]
    assert train.main(base + ["--work-dir", str(tmp_path / "w"), "--results-dir", str(tmp_path / "r"), "--mark-dry"]) == 0
    (run,) = list((tmp_path / "r").iterdir())
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert run.name.startswith("DRY_") and m["dry_run"] is True and m["folds_verified"] is True
    assert m["folds_csv_sha256"] == io_utils.sha256_file(fp) and m["cache_location"] == "other"
    assert train.main(base + ["--work-dir", str(tmp_path / "w2"), "--results-dir", str(tmp_path / "r2")]) == 0
    (run2,) = list((tmp_path / "r2").iterdir())
    assert not run2.name.startswith("DRY_")  # without the flag the id is a real one
    # the folds hash is still enforced: a wrong sidecar is refused even with --mark-dry
    io_utils.atomic_write_json(folds.sidecar_path(fp), {"folds_csv_sha256": "0" * 64})
    with pytest.raises(folds.FoldsVerificationError):
        train.main(base + ["--work-dir", str(tmp_path / "w3"), "--results-dir", str(tmp_path / "r3"), "--mark-dry"])
