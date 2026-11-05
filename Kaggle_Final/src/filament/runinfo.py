"""What a run records about itself (Gate 1b review B1; Arch section 4.5 and prompt section 5 train.py).

Everything here is read from the running environment, never typed in.  Nothing returned contains an absolute path, a
user name or a host name (review Q13: results may be committed publicly): only versions, the commit and the GPU name.
"""
from __future__ import annotations

import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

_PACKAGES = ["torch", "torchvision", "segmentation_models_pytorch", "timm", "numpy", "scipy", "opencv-python-headless",
             "opencv-python", "scikit-learn", "pycocotools", "pillow", "PyYAML", "huggingface_hub", "safetensors"]


def _version(name: str):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def git_info(repo_root: str | Path | None = None) -> dict:
    """Commit and whether the working tree differs from it.  Falls back to a ``COMMIT`` file or ``FILAMENT_COMMIT``
    (a code snapshot such as the Kaggle dataset has no ``.git``)."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    commit, dirty = None, None
    try:
        r = subprocess.run(["git", "--no-optional-locks", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            commit = r.stdout.strip()
            s = subprocess.run(["git", "--no-optional-locks", "-C", str(root), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=60)
            dirty = bool(s.stdout.strip()) if s.returncode == 0 else None
    except Exception:
        pass
    if commit is None:
        f = root / "COMMIT"
        if f.exists():
            commit = f.read_text(encoding="utf-8").strip() or None
        else:
            commit = os.environ.get("FILAMENT_COMMIT") or None
    return {"git_commit": commit, "git_dirty": dirty}


def environment_info() -> dict:
    import torch

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),  # e.g. "Windows-11-10.0.26200-SP0": no host name
        "packages": {p: _version(p) for p in _PACKAGES if _version(p)},
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),  # Phase 2.1: part of every run record
        "gpu_count": n_gpu,  # GPUs visible to this process (one per process when CUDA_VISIBLE_DEVICES is set)
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(n_gpu)],
    }


def gpu_info(device: str) -> dict:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        idx = torch.device(device).index or 0
        p = torch.cuda.get_device_properties(idx)
        return {"gpu_name": p.name, "gpu_total_gib": round(p.total_memory / 2**30, 2),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    return {"gpu_name": "cpu", "gpu_total_gib": None, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def run_id_lr(lr: float) -> str:
    """1e-4 -> ``1e-4``; 3e-5 -> ``3e-5``; 1.5e-4 -> ``1.5e-4`` (a short, unambiguous token for run ids)."""
    m, e = ("%.2e" % lr).split("e")
    return "%ge%d" % (float(m), int(e))


def make_run_id(experiment: str, resolution: int, lr: float, seed: int, dry_run: bool = False) -> str:
    """``E1_r512_lr1e-4_seed1`` (``DRY_`` prefix for dry runs).  The fold is a sub-folder, not part of the id, so that
    two processes (two GPUs) can train different folds of the same run."""
    rid = "%s_r%d_lr%s_seed%d" % (experiment, resolution, run_id_lr(lr), seed)
    return ("DRY_" + rid) if dry_run else rid


# ---------------------------------------------------------------------------------------------- Kaggle layout helpers
KNOWN_CACHE_LOCATIONS = ("/kaggle/temp/cache", "/kaggle/working/cache")


def choose_cache_dir(work: str | Path, temp_root: str | Path = "/kaggle/temp", min_free_gib: float = 3.0,
                     name: str = "cache") -> tuple[Path, str]:
    """Where the notebook builds the cache: outside the output folder if possible (Phase 2.1 list).

    ``<temp_root>/<name>`` when ``temp_root`` exists, a file can be written in it and it has at least ``min_free_gib`` GiB
    free; otherwise ``<work>/<name>``.  Nothing is assumed about ``/kaggle/temp``: every condition is tested.  Returns
    ``(folder, reason)``; the reason is printed and recorded by the caller."""
    t = Path(temp_root)
    try:
        if not t.is_dir():
            why = "%s does not exist" % t
        else:
            target = t / name
            target.mkdir(parents=True, exist_ok=True)
            probe = target / ".write_test"
            probe.write_bytes(b"x")
            probe.unlink()
            free = shutil.disk_usage(str(t)).free / 2**30
            if free >= min_free_gib:
                return target, "%s exists, is writable and has %.1f GiB free (needs %.1f)" % (t, free, min_free_gib)
            why = "%s has only %.1f GiB free (needs %.1f)" % (t, free, min_free_gib)
    except OSError as e:
        why = "%s is not usable: %s" % (t, e)
    return Path(work) / name, "fallback to the working folder because " + why


def cache_location_label(path: str | Path | None) -> str | None:
    """What a run record says about the cache folder: one of the two Kaggle locations, else ``"other"``.

    A local path is never recorded (review Q13: no machine paths in run records)."""
    if path is None:
        return None
    p = Path(path).as_posix().rstrip("/")
    return p if p in KNOWN_CACHE_LOCATIONS else "other"


def find_hf_hub(root: str | Path) -> Path | None:
    """The Hugging Face ``hub`` folder inside ``root`` (a ``hf_cache`` folder or the dataset that holds one).

    Kaggle unpacks ``hf_cache.zip`` as ``hf_cache/hf_cache/hub/models--...`` (the zip's own top folder below a folder
    named after the zip), so the folder cannot be assumed at a fixed depth (Gate 2 finding G2-13).  The shallowest folder
    called ``hub`` that contains a ``models--*`` folder wins; ``None`` if there is none."""
    root = Path(root)
    best: tuple[int, str, Path] | None = None
    candidates = [root] if root.name == "hub" else []
    candidates += [p for p in root.rglob("hub") if p.is_dir()]
    for p in candidates:
        if any(c.is_dir() and c.name.startswith("models--") for c in p.iterdir()):
            key = (len(p.relative_to(root).parts) if p != root else 0, p.as_posix())
            if best is None or key < best[:2]:
                best = (key[0], key[1], p)
    return best[2] if best else None
