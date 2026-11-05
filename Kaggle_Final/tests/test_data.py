"""Acceptance tests 1, 2 and 3 (E0): data counts, rasterisation, normalisation (Spec S3, S5; Arch P1, P2, P7, P8).

Plus unit tests of the dataset, sampler and area-downscale code that need no competition data.
"""
import json
from collections import Counter

import cv2
import numpy as np
import pytest
import torch
from pycocotools import mask as M

from filament import data

# ------------------------------------------------------------------ unit tests (synthetic arrays)


def test_area_downscale2_equals_cv2_inter_area_and_rounds_half_up():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    assert np.array_equal(data.area_downscale2(a), cv2.resize(a, (32, 32), interpolation=cv2.INTER_AREA))
    b = np.array([[0, 0], [0, 2]], np.uint8)  # mean 0.5 -> rounds to 1
    assert data.area_downscale2(b)[0, 0] == 1
    f = rng.random((8, 8)).astype(np.float32)
    assert np.allclose(data.area_downscale2(f), f.reshape(4, 2, 4, 2).mean((1, 3)))


def test_target_quantisation_error_is_below_half_a_level():
    t = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    q = data.target_to_uint8(t).astype(np.float32) / 255.0
    assert np.abs(q - t).max() <= 0.5 / 255 + 1e-7
    assert data.target_to_uint8(np.array([0, 1 / 3, 0.5, 2 / 3, 1.0], np.float32)).tolist() == [0, 85, 128, 170, 255]


def test_soft_target_levels_are_means_of_version_masks():
    a = np.array([[1, 0], [1, 1]], np.uint8)
    b = np.array([[1, 1], [0, 1]], np.uint8)
    c = np.array([[1, 0], [0, 1]], np.uint8)
    t = data.soft_target_levels([a, b, c])
    assert np.allclose(t, [[1, 1 / 3], [1 / 3, 1]])
    assert np.allclose(data.soft_target_levels([a]), a)


def test_normalise_is_p1():
    x = np.array([[0, 255]], np.uint8)
    out = data.normalise(x, 0.32742, 0.22230)
    assert np.allclose(out, [[(0 - 0.32742) / 0.22230, (1 - 0.32742) / 0.22230]], atol=1e-6)
    assert out.dtype == np.float32


def _toy(n=6, side=64):
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, (n, side, side), dtype=np.uint8)
    tgts = rng.integers(0, 256, (n, side, side), dtype=np.uint8)
    return imgs, tgts


def test_sampler_is_deterministic_and_counts_steps_not_epochs():
    kw = dict(indices=[0, 1, 2, 3, 4], batch_size=2, n_steps=7, resolution=64, crop_sides=[32, 48, 64],
              augment=data.AugmentConfig(flip=True, rot90=True, brightness=(0.9, 1.1)), seed=5)
    a = list(data.SpecSampler(**kw))
    b = list(data.SpecSampler(**kw))
    assert a == b and len(a) == 7 == len(data.SpecSampler(**kw))
    assert list(data.SpecSampler(**{**kw, "seed": 6})) != a
    for batch in a:
        assert len({s.side for s in batch}) == 1  # one crop side per batch so samples can be stacked
        for s in batch:
            assert s.side in (32, 48, 64) and 0 <= s.y0 <= 64 - s.side and 0 <= s.x0 <= 64 - s.side
            assert 0.9 <= s.brightness <= 1.1 and s.rot90 in range(4)
    # a pass over the data uses each image once: first 5 draws are a permutation of the indices
    flat = [s.index for batch in a for s in batch]
    assert sorted(flat[:5]) == [0, 1, 2, 3, 4]


def test_default_augmentation_is_none():
    batches = list(data.SpecSampler([0, 1, 2], 3, 3, 64, None, data.AugmentConfig(), seed=0))
    for batch in batches:
        for s in batch:
            assert (s.side, s.y0, s.x0, s.flip_h, s.flip_v, s.rot90, s.brightness) == (None, 0, 0, False, False, 0, 1.0)


def test_crop_sides_are_multiples_of_32_inside_the_documented_range():
    sides = data.crop_sides_for(1024, 512, 768)
    assert sides == [512, 544, 576, 608, 640, 672, 704, 736, 768]
    assert data.crop_sides_for(512, None, None) is None
    with pytest.raises(ValueError):
        data.crop_sides_for(512, 512, 768)


def test_dataset_applies_the_same_geometry_to_image_and_target_and_normalises():
    imgs, tgts = _toy()
    ds = data.FilamentDataset(imgs, tgts, mean=0.3, std=0.2)
    # whole image, no augmentation
    x, y = ds[data.SampleSpec(index=2, side=None)]
    assert x.shape == y.shape == (1, 64, 64) and x.dtype == y.dtype == torch.float32
    assert np.allclose(x[0].numpy(), (imgs[2] / 255.0 - 0.3) / 0.2, atol=1e-6)
    assert np.allclose(y[0].numpy(), tgts[2] / 255.0, atol=1e-7)
    # crop + flips + rotation: undo them and compare with the raw arrays
    spec = data.SampleSpec(index=1, side=32, y0=5, x0=9, flip_h=True, flip_v=True, rot90=1, brightness=1.0)
    x, y = ds[spec]
    assert x.shape == (1, 32, 32)
    raw = imgs[1][5:37, 9:41]
    ref = np.rot90(raw[::-1, ::-1], 1)
    assert np.allclose(x[0].numpy(), (ref / 255.0 - 0.3) / 0.2, atol=1e-6)
    assert np.allclose(y[0].numpy(), np.rot90(tgts[1][5:37, 9:41][::-1, ::-1], 1) / 255.0, atol=1e-7)
    # brightness multiplies the 0..1 intensity and clips; the target is untouched
    xb, yb = ds[data.SampleSpec(index=1, side=None, brightness=1.1)]
    assert np.allclose(xb[0].numpy(), (np.clip(imgs[1] / 255.0 * 1.1, 0, 1) - 0.3) / 0.2, atol=1e-6)
    assert np.allclose(yb[0].numpy(), tgts[1] / 255.0, atol=1e-7)


def test_dataloader_batches_have_one_crop_size_and_match_the_dataset():
    imgs, tgts = _toy(8, 128)
    ds = data.FilamentDataset(imgs, tgts, 0.3, 0.2)
    sampler = data.SpecSampler(range(8), 4, 3, 128, [64, 96], data.AugmentConfig(flip=True), seed=1)
    batches = list(sampler)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=data.SpecSampler(range(8), 4, 3, 128, [64, 96],
                                                                         data.AugmentConfig(flip=True), seed=1))
    for specs, (x, y) in zip(batches, dl):
        assert x.shape == (4, 1, specs[0].side, specs[0].side) == tuple(y.shape)
        assert torch.equal(x[0], ds[specs[0]][0])


def test_eval_dataset_returns_whole_normalised_images():
    imgs, _ = _toy(3, 64)
    ev = data.EvalDataset(imgs, 0.3, 0.2)
    x, i = ev[1]
    assert i == 1 and x.shape == (1, 64, 64)
    assert np.allclose(x[0].numpy(), (imgs[1] / 255.0 - 0.3) / 0.2, atol=1e-6)


# ------------------------------------------------------------------ test 1: counts (needs the data)
@pytest.mark.data
def test_data_counts(layout, annotations):
    J = json.loads(layout.train_json.read_text(encoding="utf-8"))
    train_files = data.list_image_names(layout.train_images)
    test_files = data.list_image_names(layout.test_images)
    per_image = Counter(len(v) for v in annotations.versions_of.values())
    n_by_inst_train = Counter(data.instrument_of(n) for n in train_files)
    n_by_inst_test = Counter(data.instrument_of(n) for n in test_files)
    ann_ids = [a["id"] for a in J["annotations"]]
    print("\n[E0-1] train images %d | test images %d | versions %d | filaments %d" % (
        len(train_files), len(test_files), annotations.n_versions, annotations.n_filaments))
    print("[E0-1] versions per image: %s | annotation ids unique: %s" % (
        dict(sorted(per_image.items())), len(set(ann_ids)) == len(ann_ids)))
    print("[E0-1] train/test name overlap: %d | JSON names == files on disk: %s" % (
        len(set(train_files) & set(test_files)), set(train_files) == set(annotations.image_names)))
    print("[E0-1] instruments train %s | test %s" % (dict(sorted(n_by_inst_train.items())),
                                                    dict(sorted(n_by_inst_test.items()))))
    assert (len(train_files), len(test_files)) == (707, 180)
    assert annotations.n_versions == 1154 and annotations.n_filaments == 8199
    assert dict(per_image) == {1: 411, 2: 145, 3: 151}
    assert not set(train_files) & set(test_files)
    assert set(train_files) == set(annotations.image_names)
    assert len(set(ann_ids)) == len(ann_ids) == 8199
    assert all(len(v) >= 1 for v in annotations.version_anns.values())
    assert dict(n_by_inst_train) == {"Bh": 118, "Ch": 125, "Lh": 127, "Mh": 125, "Th": 110, "Uh": 102}
    assert dict(n_by_inst_test) == {"Bh": 33, "Ch": 33, "Lh": 34, "Mh": 26, "Th": 27, "Uh": 27}
    # every filament is one polygon (S3)
    assert Counter(len(a["segmentation"]) for v in annotations.version_anns.values() for a in v) == {1: 8199}


# ------------------------------------------------------------------ test 2: rasterisation (needs the data)
@pytest.mark.data
def test_pycocotools_area_equals_json_area(annotations):
    diffs = []
    for anns in annotations.version_anns.values():
        for a in anns:
            diffs.append(abs(float(M.area(data.annotation_rle(a))) - a["area"]))
    diffs = np.array(diffs)
    print("\n[E0-2] pycocotools mask area vs JSON area over %d filaments: largest difference %.1f, "
          "filaments with any difference %d" % (len(diffs), diffs.max(), int((diffs > 0).sum())))
    assert len(diffs) == 8199 and diffs.max() == 0.0


@pytest.mark.data
def test_soft_target_levels_in_the_cache(cache_dir):
    cache = data.load_cache(cache_dir)
    idx = cache.meta
    levels = {}
    for key, n in idx["level_hist_2048"].items():
        k, c = (int(x) for x in key.split(":"))
        levels[round(c / k, 4)] = levels.get(round(c / k, 4), 0) + n
    total = sum(levels.values())
    print("\n[E0-2] soft-target values at 2048 px (value: pixels, share):")
    for v in sorted(levels):
        print("   %.4f: %13d  %.5f" % (v, levels[v], levels[v] / total))
    assert sorted(levels) == [0.0, 0.3333, 0.5, 0.6667, 1.0]  # Arch P7
    assert total == 707 * 2048 * 2048

    counts = np.zeros(256, np.int64)
    for i in range(len(cache.targets_1024)):
        counts += np.bincount(np.asarray(cache.targets_1024[i]).ravel(), minlength=256)
    nonzero = np.nonzero(counts)[0]
    print("[E0-2] target values at 1024 px (uint8 = round(255 x level)): %d distinct values, %.5f of pixels are 0, "
          "%.5f are 255" % (len(nonzero), counts[0] / counts.sum(), counts[255] / counts.sum()))
    top = np.argsort(-counts)[:8]
    print("   most frequent values: " + ", ".join("%d (%.5f)" % (v, counts[v] / counts.sum()) for v in sorted(top)))
    between = counts[1:255].sum() / counts.sum()
    print("   share of pixels strictly between 0 and 255 (boundaries and disagreement): %.5f" % between)
    assert counts.sum() == 707 * 1024 * 1024
    assert counts[85] > 0 and counts[128] > 0 and counts[170] > 0  # the pure levels 1/3, 1/2, 2/3 survive averaging


@pytest.mark.data
def test_cache_matches_a_fresh_computation(cache_dir, layout, annotations):
    cache = data.load_cache(cache_dir)
    assert cache.train_names == annotations.image_names
    assert cache.images_1024.shape == (707, 1024, 1024) and cache.images_1024.dtype == np.uint8
    assert cache.test_images_1024.shape == (180, 1024, 1024)
    assert cache.test_names == data.list_image_names(layout.test_images)
    for i in (0, 123, 706):
        name = cache.train_names[i]
        g = data.read_gray(data.image_path(layout.train_images, name))
        assert np.array_equal(cache.images_1024[i], cv2.resize(g, (1024, 1024), interpolation=cv2.INTER_AREA))
        t = data.target_to_uint8(data.area_downscale2(data.soft_target(annotations, name)))
        assert np.array_equal(cache.targets_1024[i], t)
        assert cache.n_versions[i] == len(annotations.versions_of[name])
    imgs512, tg512 = cache.arm(512)
    assert imgs512.shape == (707, 512, 512) and tg512.shape == (707, 512, 512)
    assert np.array_equal(imgs512[5], cv2.resize(np.asarray(cache.images_1024[5]), (512, 512), interpolation=cv2.INTER_AREA))


@pytest.mark.data
def test_normalisation_constants_match_the_documents(cache_dir):
    idx = data.load_cache(cache_dir).meta
    print("\n[E0-3] normalisation over %d pixels at 1024 px: mean %.5f std %.5f (document 0.32742, 0.22230); "
          "difference %.2e, %.2e" % (idx["n_pixels"], idx["mean_1024"], idx["std_1024"],
                                     idx["mean_1024"] - 0.32742, idx["std_1024"] - 0.22230))
    assert idx["n_pixels"] == 741343232
    assert round(idx["mean_1024"], 5) == 0.32742 and round(idx["std_1024"], 5) == 0.22230
