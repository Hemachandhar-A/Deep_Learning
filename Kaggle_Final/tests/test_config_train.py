"""config.py layering and the schedule / step loop of train.py (Arch section 4.3)."""
import math

import numpy as np
import pytest
import torch

from filament import config, data, losses, model, train


def test_config_layers_and_save_round_trip(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("train:\n  seed: 1\n  lr: 0.0001\nmodel:\n  encoder: resnet34\n", encoding="utf-8")
    exp = tmp_path / "e1b.yaml"
    exp.write_text("base_config: base.yaml\ntrain:\n  seed: 2\n", encoding="utf-8")
    cfg = config.load_config(exp)
    assert cfg["train"] == {"seed": 2, "lr": 0.0001}  # E1b differs from E1 only in the training seed
    assert cfg["model"]["encoder"] == "resnet34"
    assert config.get(cfg, "train.seed") == 2 and config.get(cfg, "train.nope", "x") == "x"
    out = tmp_path / "run" / "config.yaml"
    config.save_config(cfg, out)
    assert config.load_config(out) == cfg
    assert config.load_config(exp, overrides={"train": {"seed": 3}})["train"]["seed"] == 3


def test_repo_base_config_loads_and_builds_the_dataclasses():
    from pathlib import Path

    from filament import postprocess

    cfg = config.load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
    assert cfg["normalisation"] == {"mean": 0.32742, "std": 0.22230}
    assert cfg["folds"]["n_splits"] == 5 and cfg["optim"]["warmup_steps"] == 200
    assert cfg["folds"]["strategy"] == "balanced" and cfg["folds"]["seed"] == 42
    assert cfg["train"]["steps"] == 7000  # the approved step budget (reviewer, 2026-10-15): one place, base.yaml
    model.ModelConfig.from_dict(cfg["model"])
    losses.LossConfig.from_dict(cfg["loss"])
    train.OptimConfig.from_dict(cfg["optim"])
    postprocess.PostprocessConfig.from_dict(cfg["postprocess"])
    data.AugmentConfig(**{k: v for k, v in cfg["augment"].items()})


def test_e1_and_e1b_differ_only_in_the_training_seed_and_carry_the_decisions():
    """Gate 1 decisions: balanced folds seed 42, Adam wd 0, no augmentation, lr grid in E1; the lr check adopted 1e-3 (reviewer, 2026-10-17)."""
    from pathlib import Path

    d = Path(__file__).resolve().parents[1] / "configs"
    e1, e1b = config.load_config(d / "e1_seed1.yaml"), config.load_config(d / "e1b_seed2.yaml")
    assert e1["train"]["seed"] == 1 and e1b["train"]["seed"] == 2
    strip = lambda c: {k: v for k, v in c.items() if k not in ("experiment", "lr_check")}  # noqa: E731
    a, b = strip(e1), strip(e1b)
    a["train"], b["train"] = {k: v for k, v in a["train"].items() if k != "seed"}, {k: v for k, v in b["train"].items() if k != "seed"}
    assert a == b  # everything else, including the optimiser and the folds, is identical
    assert e1["lr_check"] == {"default": 1.0e-4, "grid": [3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3]}
    assert "lr_check" not in e1b and e1["optim"]["lr"] == 1.0e-3
    for c in (e1, e1b):
        assert c["folds"]["strategy"] == "balanced" and c["folds"]["seed"] == 42 and c["folds"]["n_splits"] == 5
        assert c["optim"]["optimizer"] == "adam" and c["optim"]["weight_decay"] == 0.0
        assert c["augment"] == {"flip": False, "rot90": False, "brightness": None}
        assert c["postprocess"]["r_max"] is None and c["postprocess"]["c_limb"] is None  # disc deferred (D-f)
        assert c["train"]["resolution"] == 512


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = prefix + "." + str(k) if prefix else str(k)
        out.update(_flat(v, key) if isinstance(v, dict) else {key: v})
    return out


# Phase 3a: each arm config differs from E1 in exactly the keys named here (and only there), with exactly these values.
ARM_DIFFERENCES = {
    "e1c_seed3.yaml": {"experiment": "E1c", "train.seed": 3},
    "e6a_flip_rot90.yaml": {"experiment": "E6a", "augment.flip": True, "augment.rot90": True},
    "e6b_flip_rot90_brightness.yaml": {"experiment": "E6b", "augment.flip": True, "augment.rot90": True,
                                       "augment.brightness": [0.9, 1.1]},
    "e4a_focal_dice.yaml": {"experiment": "E4a", "loss.bce": "focal", "loss.focal_gamma": 2.0},
    "e4b_tversky.yaml": {"experiment": "E4b", "loss.region": "tversky", "loss.tversky_alpha": 0.7, "loss.tversky_beta": 0.3},
}


@pytest.mark.parametrize("name", sorted(ARM_DIFFERENCES))
def test_arm_configs_differ_from_e1_only_in_the_named_keys(name):
    from pathlib import Path

    d = Path(__file__).resolve().parents[1] / "configs"
    e1 = {k: v for k, v in config.load_config(d / "e1_seed1.yaml").items() if k != "lr_check"}  # lr_check: record of the finished check
    arm = config.load_config(d / name)
    assert "lr_check" not in arm
    fe, fa = _flat(e1), _flat(arm)
    differing = {k: fa.get(k, "<absent>") for k in set(fe) | set(fa) if fe.get(k, "<absent>") != fa.get(k, "<absent>")}
    assert differing == ARM_DIFFERENCES[name]
    # what every arm shares with E1: the chosen lr, the step budget, the folds, the model, the post-processing
    assert arm["optim"]["lr"] == 1.0e-3 and arm["train"]["steps"] == 7000 and arm["train"]["resolution"] == 512
    assert arm["folds"] == e1["folds"] and arm["model"] == e1["model"] and arm["postprocess"] == e1["postprocess"]
    # and it builds: the training settings, the loss and the augmentation take the values (no key is silently ignored)
    s = train.settings_from_cfg(arm)
    assert s.steps == 7000 and s.seed == arm["train"]["seed"]
    losses.CombinedLoss(losses.LossConfig.from_dict(arm.get("loss")))
    assert s.augment.flip == arm.get("augment", {}).get("flip", False)


def test_tversky_with_the_neutral_weights_is_the_dice_loss_so_the_e4_arm_needs_other_weights():
    """alpha = beta = 0.5 (the code default) is not an arm: 1 - (tp+s)/(tp + fp/2 + fn/2 + s) equals the soft Dice loss up to
    the smoothing constant counting twice.  The proposal of e4b_tversky.yaml (0.7 / 0.3) is a different loss."""
    g = torch.Generator().manual_seed(0)
    y = (torch.rand(8 * 64 * 64, generator=g) < 0.02).float() * 0.5
    logits = torch.randn(8 * 64 * 64, generator=g) * 3.0
    dice = losses.soft_dice_loss(logits, y, 1.0)
    assert abs(float(losses.tversky_loss(logits, y, 0.5, 0.5, 1.0) - dice)) < 1e-3
    assert abs(float(losses.tversky_loss(logits, y, 0.7, 0.3, 1.0) - dice)) > 5e-3


def test_lr_schedule_is_linear_warmup_then_cosine_to_zero():
    total, warm = 1000, 200
    f = [train.lr_factor(s, total, warm) for s in range(total)]
    assert f[0] == pytest.approx(1 / 200) and f[199] == pytest.approx(1.0) and f[200] == pytest.approx(1.0)
    assert all(a <= b for a, b in zip(f[:200], f[1:200]))  # rising during warm-up
    assert all(a >= b for a, b in zip(f[200:], f[201:]))  # falling afterwards
    assert f[-1] < 1e-4 and f[600] == pytest.approx(0.5 * (1 + math.cos(math.pi * 400 / 800)))


def test_train_loop_runs_on_cpu_and_reduces_the_loss():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, (4, 64, 64), dtype=np.uint8)
    tgts = np.zeros((4, 64, 64), np.uint8)
    for i in range(4):
        tgts[i, 10 + 5 * i : 25 + 5 * i, 8:56] = 255
    ds = data.FilamentDataset(imgs, tgts, 0.3, 0.2)
    sampler = data.SpecSampler(range(4), batch_size=4, n_steps=15, resolution=64, crop_sides=None, seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler)
    net = model.build_model(model.ModelConfig(encoder_weights=None))
    out = train.train_loop(net, loader, losses.CombinedLoss(), train.OptimConfig(lr=1e-3, warmup_steps=3, amp=True),
                           device="cpu", log_every=100, log=None)
    h = out["history"]["loss"]
    assert out["steps"] == 15 and out["amp"] is False  # AMP only on CUDA
    assert h[-1] < h[0]
    probs = train.predict_probabilities(net, imgs, 0.3, 0.2, "cpu")
    assert probs.shape == (4, 64, 64) and probs.min() >= 0 and probs.max() <= 1
