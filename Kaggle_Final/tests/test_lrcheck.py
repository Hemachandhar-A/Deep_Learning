"""The rule of the E1 learning-rate check (Q12) and its config."""
from pathlib import Path

import pytest

from filament import config, lrcheck


def test_rule_adopts_only_beyond_the_seed_noise():
    pq = {(1e-4, 1): 0.400, (1e-4, 2): 0.395, (3e-5, 1): 0.380, (3e-4, 1): 0.404, (1e-3, 1): 0.390}
    d = lrcheck.decide(pq)
    assert d["noise_abs_diff"] == pytest.approx(0.005)
    assert d["best_candidate_lr"] == 3e-4 and d["margin_over_default_seed1"] == pytest.approx(0.004)
    assert d["adopt_other_lr"] is False and d["chosen_lr"] == 1e-4 and "not distinguishable" in d["statement"]
    pq[(3e-4, 1)] = 0.4101  # 0.0101 above the default's seed-1 value, more than the noise of 0.005
    d = lrcheck.decide(pq)
    assert d["adopt_other_lr"] is True and d["chosen_lr"] == 3e-4 and "adopt lr 0.0003" in d["statement"]
    pq[(3e-4, 1)] = 0.4051  # just above the noise of 0.005 -> adopt; the rule is strict ("more than")
    assert lrcheck.decide(pq)["adopt_other_lr"] is True
    pq[(3e-4, 1)] = 0.404  # inside the noise
    assert lrcheck.decide(pq)["adopt_other_lr"] is False


def test_rule_needs_both_default_seeds_and_a_candidate():
    with pytest.raises(KeyError):
        lrcheck.decide({(1e-4, 1): 0.4, (3e-4, 1): 0.41})
    with pytest.raises(KeyError):
        lrcheck.decide({(1e-4, 1): 0.4, (1e-4, 2): 0.39})


def test_protocol_config_matches_the_confirmed_q12():
    cfg = config.load_config(Path(__file__).resolve().parents[1] / "configs" / "e1_lrcheck.yaml")
    plan = lrcheck.plan(cfg["lr_check"])
    assert {(r["lr"], r["seed"]) for r in plan} == {(1e-4, 1), (1e-4, 2), (3e-5, 1), (3e-4, 1), (1e-3, 1)}
    assert all(r["fold"] == 0 for r in plan) and len(plan) == 5
    assert cfg["train"]["steps"] == 7000 and cfg["train"]["resolution"] == 512  # the approved budget, inherited (A4.3)
    assert cfg["optim"]["optimizer"] == "adam" and cfg["folds"]["strategy"] == "balanced"
    assert "more than" in cfg["lr_check"]["decision_rule"] and "best of t" in cfg["lr_check"]["score"]
