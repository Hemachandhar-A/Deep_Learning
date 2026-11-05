"""Acceptance test 9 (E0): post-processing on small synthetic maps (Arch section 5.2)."""
import numpy as np
import pytest
import torch

from filament import disc as discmod
from filament import postprocess as pp
from filament.postprocess import PostprocessConfig as Cfg

N = 64


def pmap(*rects, value=0.9, base=0.0):
    p = np.full((N, N), base, np.float32)
    for y0, y1, x0, x1 in rects:
        p[y0:y1, x0:x1] = value
    return p


def run(p, **kw):
    kw.setdefault("out_size", None)
    disc = kw.pop("disc", None)
    return pp.postprocess(p, Cfg(**kw), disc=disc)


def areas(res):
    return [c.area for c in res.kept]


# ---------------------------------------------------------------- threshold
def test_threshold_is_strictly_greater():
    p = pmap((10, 20, 10, 20), value=0.5)
    assert run(p, threshold=0.5).n_kept == 0  # p == t is not selected
    assert run(p, threshold=0.49).n_kept == 1
    p2 = pmap((10, 20, 10, 20), value=0.6)
    assert areas(run(p2, threshold=0.5)) == [100]


def test_zero_probability_is_never_selected_even_at_threshold_zero():
    assert run(np.zeros((N, N), np.float32), threshold=0.0).n_kept == 0


# ---------------------------------------------------------------- components
def test_components_use_8_connectivity():
    p = np.zeros((N, N), np.float32)
    p[5, 5] = p[6, 6] = p[7, 7] = 0.9  # diagonal chain: one 8-connected component
    p[20, 20] = 0.9  # separate
    res = run(p)
    assert sorted(areas(res)) == [1, 3]
    assert res.labels.max() == 2


def test_numbering_is_raster_order_and_contiguous():
    p = pmap((40, 45, 40, 45), (5, 10, 30, 35), (5, 9, 5, 9))
    res = run(p)
    first_pixels = [tuple(np.argwhere(res.labels == k)[0]) for k in (1, 2, 3)]
    assert first_pixels == sorted(first_pixels)
    assert set(np.unique(res.labels)) == {0, 1, 2, 3}


# ---------------------------------------------------------------- hysteresis
def test_hysteresis_keeps_low_pixels_attached_to_a_confident_core():
    p = np.zeros((N, N), np.float32)
    p[10:20, 10:30] = 0.35  # faint extent, 200 px
    p[12:15, 12:15] = 0.8  # confident core inside it
    p[40:50, 40:50] = 0.35  # faint blob with no core
    plain = run(p, threshold=0.5)
    assert areas(plain) == [9]
    hyst = run(p, hysteresis=(0.3, 0.6))
    assert areas(hyst) == [200]  # the faint extent is grown from the core; the core-less blob is dropped
    with pytest.raises(ValueError):
        run(p, hysteresis=(0.7, 0.6))


# ---------------------------------------------------------------- closing
def test_closing_rejoins_fragments_up_to_twice_the_radius():
    for gap, expected in [(2, 1), (3, 2)]:
        p = np.zeros((N, N), np.float32)
        p[20:30, 5:25] = 0.9
        p[20:30, 25 + gap : 45 + gap] = 0.9
        assert run(p).n_kept == 2
        assert run(p, closing_radius=1).n_kept == expected, gap
    assert pp.disk_kernel(1).tolist() == [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    assert pp.disk_kernel(2).sum() == 13


# ---------------------------------------------------------------- holes and overlaps
def test_holes_are_filled_and_open_rings_are_not():
    p = pmap((10, 30, 10, 30))
    p[15:25, 15:25] = 0.0  # a 10 x 10 hole inside a 20 x 20 square
    res = run(p)
    assert areas(res) == [400]  # filled
    assert areas(run(p, fill_holes=False)) == [300]
    p[15:25, 15:25] = 0.0
    p[19:21, 9:16] = 0.0  # cut a channel through the left wall (10 ring pixels): the hole now reaches the outside
    res = run(p)
    assert areas(res) == [300 - 10]  # not filled, because the background inside is no longer enclosed
    assert res.labels[20, 20] == 0  # the open region stays background


def test_overlap_after_filling_is_resolved_for_the_larger_component():
    p = pmap((10, 40, 10, 40))
    p[15:35, 15:35] = 0.0
    p[22:28, 22:28] = 0.9  # a small component sitting inside the hole of the ring
    res = run(p)
    assert res.n_kept == 1  # the inner blob is absorbed by the filled ring
    assert areas(res) == [900]
    absorbed = [c for c in res.components if c.drop_reason == "absorbed"]
    assert len(absorbed) == 1
    # with filling off both survive and nothing overlaps
    res2 = run(p, fill_holes=False)
    assert sorted(areas(res2)) == [36, 900 - 400]


def test_masks_are_pixel_disjoint_and_cover_the_labels():
    rng = np.random.default_rng(0)
    p = (rng.random((N, N)) > 0.6).astype(np.float32)
    res = run(p, closing_radius=1)
    stack = np.stack([res.mask(c.new_id) for c in res.kept]) if res.n_kept else np.zeros((0, N, N))
    assert (stack.sum(0) <= 1).all()
    assert stack.sum() == (res.labels > 0).sum()


# ---------------------------------------------------------------- filters
def test_area_filter_boundary():
    p = pmap((5, 15, 5, 15), (30, 40, 30, 40), (50, 53, 50, 53))  # areas 100, 100, 9
    assert sorted(areas(run(p, a_min=9))) == [9, 100, 100]
    assert sorted(areas(run(p, a_min=10))) == [100, 100]
    assert sorted(areas(run(p, a_min=101))) == []
    res = run(p, a_min=10)
    assert [c.drop_reason for c in res.components if not c.kept] == ["a_min"]


def test_confidence_filter_uses_mean_probability():
    p = pmap((5, 15, 5, 15), value=0.9)
    p[30:40, 30:40] = 0.55
    res = run(p, c_min=0.6)
    assert areas(res) == [100]
    assert res.kept[0].mean_prob == pytest.approx(0.9)
    assert res.kept[0].max_prob == pytest.approx(0.9)
    assert run(p, c_min=0.55).n_kept == 2  # p_mean == c_min is kept (dropped only when strictly below)


def test_disc_masking_and_limb_filter():
    d = discmod.Disc(cx=32.0, cy=32.0, radius=30.0)
    p = np.zeros((N, N), np.float32)
    p[30:34, 30:34] = 0.9  # centre, r/R ~ 0
    p[30:34, 55:59] = 0.9  # r/R ~ 0.8
    p[30:34, 60:64] = 0.9  # r/R ~ 0.95 (centroid at x = 62)
    assert run(p, disc=d).n_kept == 3
    res = run(p, disc=d, c_limb=0.9)
    assert res.n_kept == 2
    assert [c.drop_reason for c in res.components if not c.kept] == ["c_limb"]
    res = run(p, disc=d, r_max=0.9)  # zeroing beyond r/R 0.9 removes the outer blob (it never becomes a component)
    assert res.n_kept == 2 and not any(c.drop_reason for c in res.components)
    with pytest.raises(ValueError):
        run(p, c_limb=0.9)  # needs a disc


def test_image_without_survivors_has_no_masks_and_no_rows():
    res = run(pmap((5, 8, 5, 8)), a_min=1000)
    assert res.n_kept == 0 and res.rles() == [] and res.counts() == [] and res.labels.max() == 0


# ---------------------------------------------------------------- resolution, RLE and torch agreement
def test_upsampling_matches_torch_bilinear_and_dtype():
    rng = np.random.default_rng(0)
    small = rng.random((16, 16)).astype(np.float32)
    ours = pp.upsample_probability(small.astype(np.float16).astype(np.float32), 32)
    ref = torch.nn.functional.interpolate(
        torch.from_numpy(small.astype(np.float16).astype(np.float32))[None, None], size=(32, 32),
        mode="bilinear", align_corners=False)[0, 0].numpy()
    assert np.abs(ours - ref).max() < 1e-5
    assert pp.upsample_probability(small.astype(np.float16), 32).dtype == np.float32
    assert pp.upsample_probability(np.full((8, 8), 0.3, np.float32), 16).std() < 1e-7  # constants stay constant


def test_full_size_run_and_csv_counts_round_trip():
    from filament import rle

    p = np.zeros((1024, 1024), np.float32)
    p[100:140, 100:300] = 0.9
    p[500:520, 600:640] = 0.8
    res = pp.postprocess(p, Cfg(threshold=0.5, a_min=500))
    assert res.labels.shape == (2048, 2048)
    counts = res.counts()
    assert len(counts) == res.n_kept == 2
    for c, k in zip(counts, (1, 2)):
        assert np.array_equal(rle.decode_counts(c), res.mask(k))
    # upsampled blobs are about 4x the 1024 px area (2 x 2 per side), not exactly because of the bilinear edge
    assert 0.9 * 4 * 8000 < res.kept[0].area < 1.1 * 4 * 8000
