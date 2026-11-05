"""Acceptance test 7 (E0): RLE and CSV (Spec S4).  Unit tests on synthetic masks plus a data test."""
import numpy as np
import pytest

from filament import rle

H = W = 2048


def _blob(y0, y1, x0, x1):
    m = np.zeros((H, W), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def test_encode_decode_round_trip_synthetic():
    rng = np.random.default_rng(0)
    for _ in range(5):
        m = (rng.random((H, W)) > 0.995).astype(np.uint8)
        s = rle.encode_mask(m)
        assert isinstance(s, str) and s
        assert np.array_equal(rle.decode_counts(s), m)


def test_encode_masks_matches_encode_mask_and_decode_many():
    ms = np.stack([_blob(10, 50, 10, 90), _blob(100, 130, 500, 700), _blob(1000, 1001, 5, 2000)])
    strs = rle.encode_masks(ms)
    assert strs == [rle.encode_mask(m) for m in ms]
    assert np.array_equal(rle.decode_many(strs), ms)
    assert rle.encode_masks(np.zeros((0, H, W), np.uint8)) == []
    assert rle.decode_many([]).shape == (0, H, W)


def test_fortran_order_is_used():
    """A mask that is non-zero only in the first row: column-major RLE starts with many short runs, not one run."""
    m = np.zeros((H, W), np.uint8)
    m[0, :] = 1
    s_row = rle.encode_mask(m)
    s_col = rle.encode_mask(m.T.copy())  # first column only
    assert s_row != s_col
    assert np.array_equal(rle.decode_counts(s_row), m)


def test_shape_and_value_guards():
    with pytest.raises(ValueError):
        rle.encode_mask(np.zeros((64, 64), np.uint8))
    with pytest.raises(ValueError):
        rle.encode_mask(np.full((H, W), 255, np.uint8))
    assert rle.encode_mask(np.zeros((64, 64), np.uint8), expected_shape=None)


def test_overlap_counter():
    a, b = _blob(0, 10, 0, 10), _blob(5, 15, 5, 15)
    assert rle.count_overlap_pixels(np.stack([a, b])) == 25
    assert rle.count_overlap_pixels(np.stack([a, _blob(20, 30, 20, 30)])) == 0
    assert rle.count_overlap_pixels(a[None]) == 0


def _write(tmp_path, items):
    p = tmp_path / "sub.csv"
    rle.write_submission_csv(p, [(fid, rle.encode_mask(m)) for fid, m in items])
    return p


def test_csv_round_trip_and_ids(tmp_path):
    masks = {
        "20150125172714Mh_1": _blob(10, 50, 10, 90),
        "20150125172714Mh_2": _blob(100, 140, 100, 190),
        "20160301000000Bh_1": _blob(7, 9, 7, 2000),
    }
    p = _write(tmp_path, masks.items())
    header, rows = rle.read_submission_csv(p)
    assert header == ["filament_id", "segmentation_rle"]
    assert [r[0] for r in rows] == list(masks)
    rep = rle.check_submission_csv(p, expected_images={"20150125172714Mh", "20160301000000Bh", "20170101000000Th"})
    assert rep.ok, rep.summary()
    assert rep.n_rows == 3 and rep.n_images == 2
    assert rle.verify_round_trip(p, masks) == []
    # the file is plain text with no quotes: counts strings are written bare
    assert '"' not in p.read_text()
    assert rle.make_filament_ids("20150125172714Mh", 3) == [
        "20150125172714Mh_1", "20150125172714Mh_2", "20150125172714Mh_3"]


def test_engineered_overlap_is_caught(tmp_path):
    a, b = _blob(10, 50, 10, 90), _blob(40, 80, 80, 120)  # overlap rows 40-50, cols 80-90 = 10 x 10
    p = _write(tmp_path, {"20150125172714Mh_1": a, "20150125172714Mh_2": b}.items())
    rep = rle.check_submission_csv(p)
    assert not rep.ok
    assert any("100 pixels" in s for s in rep.problems), rep.problems


def test_other_problems_are_caught(tmp_path):
    good = rle.encode_mask(_blob(10, 50, 10, 90))
    # wrong header
    p = tmp_path / "bad_header.csv"
    p.write_text("id,rle\n20150125172714Mh_1,%s\n" % good)
    assert not rle.check_submission_csv(p).ok
    # bad id, duplicate id, unknown image, empty mask
    rows = [("nounderscore", good), ("20150125172714Mh_1", good), ("20150125172714Mh_1", good),
            ("99999999999999Zz_1", good), ("20150125172714Mh_2", rle.encode_mask(np.zeros((H, W), np.uint8)))]
    rep = rle.check_submission_rows(list(rle.CSV_COLUMNS), rows, expected_images={"20150125172714Mh"})
    text = " | ".join(rep.problems)
    assert "does not look like" in text and "duplicate id" in text and "not a known test image" in text
    assert "empty mask" in text


def test_image_without_predictions_has_no_rows(tmp_path):
    p = _write(tmp_path, {"20150125172714Mh_1": _blob(1, 5, 1, 5)}.items())
    header, rows = rle.read_submission_csv(p)
    assert {r[0].rsplit("_", 1)[0] for r in rows} == {"20150125172714Mh"}  # an image not predicted has no row
    empty = tmp_path / "empty.csv"
    assert rle.write_submission_csv(empty, []) == 0
    assert rle.read_submission_csv(empty) == (["filament_id", "segmentation_rle"], [])
    assert rle.check_submission_csv(empty).ok


@pytest.mark.data
def test_round_trip_on_all_training_filaments(annotations):
    """Encode and decode all 8,199 training filament masks; they must be identical (acceptance test 7)."""
    from filament import data

    n = bad = 0
    for version_id, anns in annotations.version_anns.items():
        for a in anns:
            m = data.rasterise_annotation(a)
            if not np.array_equal(rle.decode_counts(rle.encode_mask(m)), m):
                bad += 1
            n += 1
    print("\n[E0-7] RLE round trip: %d filament masks, %d mismatches" % (n, bad))
    assert n == 8199 and bad == 0


# ---------------------------------------------------------------- crop-based encoding (efficiency fix of the dry run)
def _full(y0, y1, x0, x1, shape=(H, W)):
    m = np.zeros(shape, np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def test_encode_crop_equals_encode_mask_on_random_and_edge_cases():
    rng = np.random.default_rng(5)
    cases = []
    for _ in range(60):  # random blobs of random size at random places, including touching every border
        h, w = int(rng.integers(1, 300)), int(rng.integers(1, 300))
        y0, x0 = int(rng.integers(0, H - h + 1)), int(rng.integers(0, W - w + 1))
        crop = (rng.random((h, w)) > rng.uniform(0.05, 0.95)).astype(np.uint8)
        cases.append((crop, y0, x0))
    cases += [
        (np.ones((1, 1), np.uint8), 0, 0),  # the very first pixel
        (np.ones((1, 1), np.uint8), H - 1, W - 1),  # the very last pixel: no trailing zero run
        (np.ones((H, W), np.uint8), 0, 0),  # everything
        (np.zeros((5, 5), np.uint8), 7, 9),  # an empty crop
        (np.ones((H, 1), np.uint8), 0, 0),  # exactly one column
        (np.ones((1, W), np.uint8), 0, 0),  # exactly one row: 2048 runs of length 1
        (np.ones((3, 3), np.uint8), H - 3, W - 3),
    ]
    for crop, y0, x0 in cases:
        full = np.zeros((H, W), np.uint8)
        full[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop
        assert rle.encode_crop(crop, y0, x0) == rle.encode_mask(full)
        assert np.array_equal(rle.decode_counts(rle.encode_crop(crop, y0, x0)), full)
    with pytest.raises(ValueError):
        rle.encode_crop(np.ones((3, 3), np.uint8), H - 2, 0)


def test_overlap_is_detected_without_stacking_and_counted_exactly(tmp_path):
    # three masks: A and B overlap in 40 x 10 = 400 px, C is disjoint; the count is reported exactly
    a, b, c = _full(100, 140, 100, 200), _full(100, 140, 190, 300), _full(500, 520, 500, 520)
    p = _write(tmp_path, {"20150125172714Mh_1": a, "20150125172714Mh_2": b, "20150125172714Mh_3": c}.items())
    rep = rle.check_submission_csv(p)
    assert not rep.ok and any("400 pixels" in s for s in rep.problems)
    ok = _write(tmp_path, {"20150125172714Mh_1": a, "20150125172714Mh_2": c}.items())
    assert rle.check_submission_csv(ok).ok
    # many disjoint masks must not need memory proportional to their number
    many = {"20150125172714Mh_%d" % (k + 1): _full(10 * (k // 40), 10 * (k // 40) + 5, 50 * (k % 40), 50 * (k % 40) + 5)
            for k in range(400)}
    assert rle.check_submission_csv(_write(tmp_path, many.items())).ok
