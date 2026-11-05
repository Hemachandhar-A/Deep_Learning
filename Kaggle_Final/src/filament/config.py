"""Configuration files (Arch section 4.8 item 3 and 5: ``data_root`` and all seeds live in the config).

One YAML or JSON file per experiment, layered on ``configs/base.yaml``.  Nothing here knows about paths of a particular
machine: ``data_root`` and ``work_dir`` are given in the experiment file or on the command line.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from .io_utils import atomic_write_text


def _load_one(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    cfg = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise ValueError("%s must contain a mapping at the top level" % path)
    return cfg


def deep_update(base: dict, new: dict) -> dict:
    """Return ``base`` updated by ``new``; nested mappings are merged, everything else replaced."""
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, base: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load ``path``; if ``base`` is given (or the file names one under ``base_config``), layer onto it."""
    path = Path(path)
    cfg = _load_one(path)
    base_path = base or cfg.pop("base_config", None)
    if base_path is not None:
        bp = Path(base_path)
        if not bp.is_absolute():
            bp = path.parent / bp
        cfg = deep_update(load_config(bp), cfg)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def load_with_local(path: str | Path, local: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Experiment config, then the per-machine ``configs/local*.yaml`` (git-ignored: paths, the expected folds hash),
    then command-line overrides.  Later layers win; nested mappings are merged."""
    cfg = load_config(path)
    if local is not None:
        cfg = deep_update(cfg, _load_one(Path(local)))
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    """Write the fully resolved config next to a run's results (so a run can be repeated from its folder)."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        atomic_write_text(path, json.dumps(cfg, indent=1, sort_keys=True))
    else:
        atomic_write_text(path, yaml.safe_dump(cfg, sort_keys=False))


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """``get(cfg, "train.lr")``; returns ``default`` if a key is missing."""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
