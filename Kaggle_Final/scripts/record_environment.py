"""Record the software environment of the machine that ran the tests (acceptance test 13; reproducibility O5).

    python scripts/record_environment.py [--out environment.json]

Prints and optionally writes: Python, OS, torch (+CUDA), segmentation-models-pytorch, pycocotools, numpy, OpenCV,
scikit-learn, timm and the other packages the code imports, the GPU, the git commit, and the revision of the cached
ImageNet weights.  Versions are read from package metadata, never guessed.
"""
import _bootstrap  # noqa: F401

import argparse
import importlib.metadata as md
import json
import platform
import subprocess
import sys
from pathlib import Path

PACKAGES = [
    "torch", "torchvision", "segmentation_models_pytorch", "timm", "huggingface_hub", "safetensors", "pycocotools",
    "numpy", "scipy", "opencv-python", "opencv-python-headless", "scikit-learn", "pillow", "pyyaml", "tqdm", "pytest",
    "matplotlib", "pandas",
]


def pkg_version(name: str):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    import torch

    env = {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {p: pkg_version(p) for p in PACKAGES},
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            pr = torch.cuda.get_device_properties(i)
            env["gpus"].append({"name": pr.name, "total_memory_gib": round(pr.total_memory / 2**30, 2),
                                "capability": "%d.%d" % (pr.major, pr.minor)})
    try:
        import cv2

        env["cv2_imported_version"] = cv2.__version__
    except Exception as exc:  # pragma: no cover
        env["cv2_imported_version"] = "import failed: %s" % exc
    try:
        from pycocotools import mask  # noqa: F401

        env["pycocotools_import"] = "ok"
    except Exception as exc:  # pragma: no cover
        env["pycocotools_import"] = "FAILED: %s" % exc
    try:
        from filament.model import pretrained_weights_info

        env["resnet34_imagenet_weights"] = pretrained_weights_info("resnet34", "imagenet")
    except Exception:
        env["resnet34_imagenet_weights"] = None
    try:
        env["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                           cwd=Path(__file__).resolve().parents[1]).stdout.strip() or None
    except Exception:
        env["git_commit"] = None
    text = json.dumps(env, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
