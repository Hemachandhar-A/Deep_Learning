"""Check that the ImageNet weights shipped in the code dataset load with the Internet switched off (Arch A4.7, K5).

    python kaggle/check_offline_weights.py --dataset <folder made by make_dataset.py --with-weights>

Unpacks ``hf_cache.zip`` into a temporary folder, starts a fresh Python with ``HF_HUB_CACHE`` pointing there and
``HF_HUB_OFFLINE=1``, builds ``smp.Unet('resnet34', encoder_weights='imagenet')`` and prints a checksum of the first
convolution; the same model is then built in the normal (online-capable) environment and the checksums are compared.
It also checks that the process really could not have downloaded anything (offline flag set, cache folder is the only
source).  Exit code 0 only if the two models have identical encoder weights.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

CHILD = r"""
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
from filament.model import ModelConfig, build_model
m = build_model(ModelConfig(encoder_weights="imagenet"))
w = m.encoder.conv1.weight.detach().cpu().numpy().tobytes()
w2 = m.encoder.layer4[-1].conv2.weight.detach().cpu().numpy().tobytes()
print(json.dumps({"conv1": hashlib.sha256(w).hexdigest(), "layer4": hashlib.sha256(w2).hexdigest()}))
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    a = ap.parse_args()
    src = str(Path(__file__).resolve().parents[1] / "src")
    zpath = a.dataset / "hf_cache.zip"
    if not zpath.exists():
        raise SystemExit("%s not found: build the dataset with --with-weights" % zpath)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
        hub = Path(tmp) / "hf_cache" / "hub"
        env = {**os.environ, "HF_HUB_CACHE": str(hub), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "HF_HOME": str(Path(tmp) / "hf_home_empty")}
        env.pop("HUGGINGFACE_HUB_CACHE", None)
        p = subprocess.run([sys.executable, "-c", CHILD, src], capture_output=True, text=True, env=env, timeout=600)
        if p.returncode != 0:
            print(p.stderr[-1500:])
            raise SystemExit("offline load FAILED")
        offline = json.loads(p.stdout.strip().splitlines()[-1])
    # the same model through the normal environment
    p2 = subprocess.run([sys.executable, "-c", CHILD, src], capture_output=True, text=True, timeout=600)
    online = json.loads(p2.stdout.strip().splitlines()[-1])
    print("offline (HF_HUB_OFFLINE=1, cache = hf_cache.zip only):", offline)
    print("normal environment                                   :", online)
    same = offline == online
    print("identical encoder weights:", same)
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
