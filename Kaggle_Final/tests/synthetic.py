"""Synthetic helpers shared by the train / infer / pipeline tests: a tiny cache of 64 px images with one dark bar each,
a config dict in the shape of the YAML files, and dry-run folds.  CPU only, random-weight U-Net."""
import numpy as np

from filament import data, folds


def make_cache(n=12, side=64, seed=0):
    rng = np.random.default_rng(seed)
    names = ["2015%010d%s" % (i, ("Bh", "Ch", "Lh")[i % 3]) for i in range(n)]
    imgs = rng.integers(20, 200, (n, side, side), dtype=np.uint8)
    tgts = np.zeros((n, side, side), np.uint8)
    for i in range(n):
        tgts[i, 10 + (i % 4) * 8 : 18 + (i % 4) * 8, 6 : side - 6] = 255
        imgs[i] = np.where(tgts[i] > 0, imgs[i] // 2, imgs[i])  # filaments are darker
    return data.Cache(train_names=names, test_names=[], n_versions=[1 + i % 3 for i in range(n)],
                      images_1024=imgs, targets_1024=tgts, test_images_1024=None, meta={})


def make_cfg(**over):
    cfg = {
        "experiment": "T",
        "train": {"resolution": 512, "batch_size": 2, "steps": 6, "seed": 1, "checkpoint_every": 2, "num_workers": 0},
        "optim": {"optimizer": "adam", "lr": 1e-3, "weight_decay": 0.0, "warmup_steps": 2, "amp": False},
        "model": {"arch": "Unet", "encoder": "resnet34", "encoder_weights": None, "in_channels": 1, "classes": 1},
        "loss": {}, "augment": {"flip": False, "rot90": False, "brightness": None},
        "normalisation": {"mean": 0.3, "std": 0.2},
        "folds": {"n_splits": 2, "strategy": "balanced", "seed": 42, "file": "should/not/be/published.csv"},
        "paths": {"data_root": "C:/secret/place", "work_dir": "C:/secret/work"},
    }
    for k, v in over.items():
        cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) else v
    return cfg


def rows_for(cache, k=2):
    return folds.round_robin_folds({n: list(range(v)) for n, v in zip(cache.train_names, cache.n_versions)}, k)


DRY_INFO = {"folds_csv_sha256": None, "expected_sha256_source": None, "folds_verified": False,
            "scikit_learn_version": "x", "folds_strategy": "round_robin_dry_run", "folds_seed": None}
