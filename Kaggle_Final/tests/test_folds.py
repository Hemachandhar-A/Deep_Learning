"""Acceptance test 4 (E0): folds (Arch section 4.1; decisions of 2026-10-12: balanced strategy, seed 42)."""
import hashlib
import warnings
from collections import Counter

import pytest

from filament import folds


def _synthetic(n_images=200, seed=0):
    # >= 200 images: the default strategy stratifies by site x number of versions (18 classes) and scikit-learn needs
    # at least n_splits members in some class; the real data have 18 or more images in every class
    import random

    rnd = random.Random(seed)
    versions_of = {}
    for i in range(n_images):
        inst = rnd.choice(["Bh", "Ch", "Lh", "Mh", "Th", "Uh"])
        name = "2015%010d%s" % (i, inst)  # 14 digits then the two-letter site code
        versions_of[name] = ["%02d-%s" % (v, name) for v in range(rnd.choice([1, 1, 2, 3]))]
        versions_of[name] = versions_of[name] or ["00-%s" % name]
    return versions_of


@pytest.fixture(autouse=True)
def _quiet_small_class_warning():
    # tiny synthetic classes (fewer than 5 members) trigger a scikit-learn UserWarning that is irrelevant here
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The least populated class")
        yield


def test_decided_defaults():
    assert folds.DEFAULT_STRATEGY == "balanced" and folds.DEFAULT_SEED == 42  # user's decision D-a, 2026-10-12


def test_synthetic_folds_are_valid_and_deterministic():
    v = _synthetic()
    a = folds.make_folds(v)  # defaults: balanced, seed 42, 5 folds
    assert a == folds.make_folds(v, 5, seed=42, strategy="balanced")
    assert a != folds.make_folds(v, 5, seed=43)
    assert folds.check_folds(a, v) == []
    assert Counter(r["fold"] for r in a).keys() == set(range(5))


def test_every_version_shares_the_fold_of_its_image(tmp_path):
    v = _synthetic()
    rows = folds.make_folds(v, 5, seed=1)
    fold_of = {r["image"]: r["fold"] for r in rows}
    # a version belongs to exactly one image, so one fold per image means no image straddles a split
    assert set(fold_of) == set(v) and all(0 <= fold_of[n] < 5 for n in v)
    p = tmp_path / "folds.csv"
    folds.write_folds_csv(p, rows)
    assert folds.read_folds_csv(p) == rows


def test_fold_indices_partition_the_cache_order():
    v = _synthetic(200)
    rows = folds.make_folds(v, 5, seed=1)
    names = sorted(v)
    seen = []
    for k in range(5):
        train, val = folds.fold_indices(rows, names, k)
        assert set(train).isdisjoint(val) and len(train) + len(val) == len(names)
        seen += val
    assert sorted(seen) == list(range(len(names)))  # each image is held out exactly once


def test_check_folds_detects_problems():
    v = _synthetic(200)
    rows = folds.make_folds(v, 5, seed=1)
    bad = [dict(r) for r in rows] + [dict(rows[0])]
    assert "an image appears more than once" in folds.check_folds(bad, v)
    bad = [dict(r) for r in rows]
    bad[0]["fold"] = 9
    assert any("has fold 9" in p for p in folds.check_folds(bad, v))


@pytest.mark.parametrize("strategy", ["document", "balanced"])
def test_both_strategies_are_valid_and_deterministic(strategy):
    v = _synthetic(200)
    a = folds.make_folds(v, 5, seed=3, strategy=strategy)
    assert a == folds.make_folds(v, 5, seed=3, strategy=strategy)
    assert folds.check_folds(a, v) == []
    with pytest.raises(ValueError):
        folds.make_folds(v, 5, seed=3, strategy="nope")


def test_file_hash_is_stable_and_verified(tmp_path):
    v = _synthetic(200)
    rows = folds.make_folds(v)
    p = tmp_path / "folds.csv"
    folds.write_folds_csv(p, rows)
    raw = p.read_bytes()
    assert b"\r" not in raw  # '\n' line endings on every platform, so the hash is the same on Windows and Linux
    assert folds.file_sha256(p) == hashlib.sha256(raw).hexdigest()
    prov = folds.provenance(p)
    import sklearn

    assert prov == {"folds_csv_sha256": hashlib.sha256(raw).hexdigest(), "scikit_learn_version": sklearn.__version__}
    assert folds.verify_folds_file(p, prov["folds_csv_sha256"]) == prov["folds_csv_sha256"]
    assert folds.verify_folds_file(p, None) == prov["folds_csv_sha256"]
    with pytest.raises(ValueError):
        folds.verify_folds_file(p, "0" * 64)


@pytest.mark.data
def test_real_folds(annotations, tmp_path):
    """The decided split on the real data: every image once, all versions with it, balanced across sites."""
    rows = folds.make_folds(annotations.versions_of)  # defaults = balanced, seed 42
    assert folds.check_folds(rows, annotations.versions_of) == []
    # no image in two folds; all versions of an image share its fold (one fold per image, versions are its children)
    assert len({r["image"] for r in rows}) == len(rows) == 707
    version_fold = {}
    for r in rows:
        for vid in annotations.versions_of[r["image"]]:
            version_fold[vid] = r["fold"]
    assert len(version_fold) == 1154
    tab = folds.fold_table(rows)
    insts = sorted({r["instrument"] for r in rows})
    sp = folds.spreads(rows)
    import sklearn

    print("\n[E0-4] folds: strategy balanced, seed 42, StratifiedKFold(5), scikit-learn %s, 707 images / 1154 versions"
          % sklearn.__version__)
    print("  fold | images versions | " + " ".join("%3s" % s for s in insts))
    for k, e in tab.items():
        print("  %4d | %6d %8d | " % (k, e["images"], e["versions"]) + " ".join("%3d" % e["by_instrument"][s] for s in insts))
    print("  worst-site image spread %(worst_site_image_spread)d, fold version-total spread %(version_total_spread)d" % sp)
    assert sp["worst_site_image_spread"] <= 2 and sp["version_total_spread"] <= 6
    sizes = [e["images"] for e in tab.values()]
    assert max(sizes) - min(sizes) <= 1
    p = tmp_path / "folds.csv"
    folds.write_folds_csv(p, rows)
    assert folds.read_folds_csv(p) == rows


@pytest.mark.data
def test_balanced_strategy_spread_bounds(annotations):
    """Over 20 seeds the balanced split has a worst-site image spread <= 2 and fold version totals within 6.

    Only the ``balanced`` bounds are asserted (they hold under scikit-learn 1.6.1 and 1.9.1, Gate 1 review F2).  The
    ``document`` numbers are printed for the record; they depend on the scikit-learn version (1.8.0 changed
    StratifiedGroupKFold), so nothing is asserted about them.
    """
    import numpy as np
    import sklearn

    doc = np.array([list(folds.spreads(folds.make_folds(annotations.versions_of, 5, s, "document")).values())
                    for s in range(20)])
    bal = np.array([list(folds.spreads(folds.make_folds(annotations.versions_of, 5, s, "balanced")).values())
                    for s in range(20)])
    print("\n[E0-4] 20 seeds, scikit-learn %s, (worst-site image spread, fold version-total spread): "
          "document median %s max %s | balanced median %s max %s"
          % (sklearn.__version__, np.median(doc, 0).tolist(), doc.max(0).tolist(), np.median(bal, 0).tolist(),
             bal.max(0).tolist()))
    assert bal[:, 0].max() <= 2 and bal[:, 1].max() <= 6
    rows = folds.make_folds(annotations.versions_of, 5, 42, "balanced")
    assert folds.check_folds(rows, annotations.versions_of) == [] and len(rows) == 707


def _expected_folds_sha256():
    """Where an expected hash may come from (review B4): git-ignored ``configs/local*.yaml`` (``folds.sha256``), the
    environment (``FILAMENT_FOLDS_SHA256``), or the sidecar next to ``FILAMENT_FOLDS_CSV``.  None if there is none."""
    import json
    import os
    from pathlib import Path

    import yaml

    for p in sorted((Path(__file__).resolve().parents[1] / "configs").glob("local*.yaml")):
        h = ((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("folds") or {}).get("sha256")
        if h:
            return h, "configs/%s" % p.name
    if os.environ.get("FILAMENT_FOLDS_SHA256"):
        return os.environ["FILAMENT_FOLDS_SHA256"], "FILAMENT_FOLDS_SHA256"
    csv_path = os.environ.get("FILAMENT_FOLDS_CSV")
    if csv_path and folds.sidecar_path(csv_path).exists():
        return json.loads(folds.sidecar_path(csv_path).read_text(encoding="utf-8"))["folds_csv_sha256"], "sidecar"
    return None, None


@pytest.mark.data
def test_regenerated_folds_have_exactly_the_expected_bytes(annotations, tmp_path):
    """Review B4: regenerate folds.csv from the annotation file and compare its SHA-256 with the expected hash.

    Skipped when no expected hash is available (set it in the git-ignored configs/local.yaml, or FILAMENT_FOLDS_SHA256,
    or FILAMENT_FOLDS_CSV pointing at the file with its sidecar)."""
    expected, source = _expected_folds_sha256()
    if expected is None:
        pytest.skip("no expected folds hash available (configs/local*.yaml, FILAMENT_FOLDS_SHA256, FILAMENT_FOLDS_CSV)")
    rows = folds.make_folds(annotations.versions_of)  # defaults: balanced, seed 42, 5 folds
    p = tmp_path / "folds.csv"
    folds.write_folds_csv(p, rows)
    got = folds.file_sha256(p)
    import sklearn

    print("\n[Phase2-B4] regenerated folds.csv sha256 %s...; expected (%s) %s...; scikit-learn %s"
          % (got[:16], source, expected[:16], sklearn.__version__))
    assert got == expected
