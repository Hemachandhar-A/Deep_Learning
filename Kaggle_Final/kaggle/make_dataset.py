"""Assemble the private Kaggle dataset that carries the code, the folds and (optionally) the ImageNet weights.

    python kaggle/make_dataset.py --folds-csv <work>/folds.csv --out <folder> [--with-weights] [--allow-dirty]

Option B of the code-delivery options (README of ``kaggle/``): no credentials, version pinned by the dataset version.

The folder it writes contains

* ``code.zip``  ``repo/`` = every file tracked by git at HEAD (code, configs, tests, text; never data, caches or
  local configs) with ``COMMIT`` and ``REPO_MANIFEST.sha256`` inside; one file, because ``kaggle datasets create``
  skips sub-folders unless told otherwise, and so that the notebook works whether Kaggle keeps or unpacks the zip;
* ``COMMIT``    the commit hash (a Kaggle copy has no ``.git``; ``runinfo.git_info`` reads this file);
* ``folds.csv`` and ``folds.csv.provenance.json``, verified before they are copied (the file lists competition file
  names, so the dataset must stay private: ``kaggle datasets create`` makes it private unless ``--public`` is given);
* ``hf_cache.zip`` with ``--with-weights``: the ResNet34 ImageNet weights laid out as a Hugging Face cache, for use
  when the Internet is off (unpack, set ``HF_HUB_CACHE`` to ``hf_cache/hub`` and ``HF_HUB_OFFLINE=1``);
* ``CODE_MANIFEST.sha256`` the SHA-256 of every file above, checked by the notebook before anything runs;
* ``dataset-metadata.json`` for ``kaggle datasets create`` (the user runs that; no token is used here).
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

from filament import folds, runinfo  # noqa: E402
from filament.model import pretrained_weights_info  # noqa: E402

DATASET_SLUG = "solar-filament-code"
OWNER = "hemachandhara"  # the user's Kaggle account name (prompt section 2); used only in metadata, never with a token


def tracked_files(repo: Path) -> list[str]:
    r = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), "ls-files"], capture_output=True, text=True, check=True)
    return [x for x in r.stdout.splitlines() if x.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def pack_zip(folder: Path, zip_path: Path, prefix: str) -> None:
    """Deterministic zip of ``folder`` (sorted names, fixed timestamps), every name starting with ``prefix/``."""
    import zipfile

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(x for x in folder.rglob("*") if x.is_file()):
            zi = zipfile.ZipInfo("%s/%s" % (prefix, p.relative_to(folder).as_posix()), date_time=(2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, p.read_bytes())


def copy_weights(out: Path) -> dict:
    """ResNet34 ImageNet weights, as the pinned revision of SMP, from the local Hugging Face cache (no download)."""
    from huggingface_hub import hf_hub_download

    info = pretrained_weights_info("resnet34", "imagenet")
    repo_id, rev = info["repo_id"], info["revision"]
    cache_name = "models--" + repo_id.replace("/", "--")
    snap = out / "hf_cache" / "hub" / cache_name / "snapshots" / rev
    snap.mkdir(parents=True, exist_ok=True)
    for fn in ("config.json", "model.safetensors"):
        src = hf_hub_download(repo_id, fn, revision=rev, local_files_only=True)  # cached copy; raises if absent
        shutil.copy2(Path(src).resolve(), snap / fn)
    refs = out / "hf_cache" / "hub" / cache_name / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(rev, encoding="utf-8")
    return {"repo_id": repo_id, "revision": rev}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds-csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--with-weights", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true", help="NOT for a dataset that a run will be reported from")
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit("%s is not empty; give a new folder" % a.out)

    gi = runinfo.git_info(REPO)
    if gi["git_commit"] is None:
        raise SystemExit("cannot read the git commit of %s" % REPO)
    if gi["git_dirty"] and not a.allow_dirty:
        raise SystemExit("the working tree differs from HEAD: commit first (the dataset must equal a commit)")
    info = folds.verify_for_run(a.folds_csv)  # refuses a file that does not match its sidecar / expected hash
    print("folds.csv verified: sha256 %s, scikit-learn %s" % (info["folds_csv_sha256"], info["scikit_learn_version"]))

    out = a.out
    stage = out / "_stage"  # the repo tree is staged here and packed into code.zip, then removed
    for rel in tracked_files(REPO):
        dst = stage / "repo" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst)
    (stage / "repo" / "COMMIT").write_text(gi["git_commit"] + "\n", encoding="utf-8", newline="\n")
    # inner manifest: every file of the tree, so the notebook can verify the code whether Kaggle kept the zip or not
    tree = sorted(p for p in (stage / "repo").rglob("*") if p.is_file())
    (stage / "repo" / "REPO_MANIFEST.sha256").write_text(
        "\n".join("%s  %s" % (sha256(p), p.relative_to(stage / "repo").as_posix()) for p in tree) + "\n",
        encoding="utf-8", newline="\n")
    pack_zip(stage / "repo", out / "code.zip", prefix="repo")
    shutil.rmtree(stage)
    (out / "COMMIT").write_text(gi["git_commit"] + "\n", encoding="utf-8", newline="\n")
    shutil.copy2(a.folds_csv, out / "folds.csv")
    shutil.copy2(folds.sidecar_path(a.folds_csv), out / "folds.csv.provenance.json")
    weights = None
    if a.with_weights:
        wstage = out / "_wstage"
        weights = copy_weights(wstage)
        pack_zip(wstage / "hf_cache", out / "hf_cache.zip", prefix="hf_cache")
        shutil.rmtree(wstage)

    files = sorted(p for p in out.rglob("*") if p.is_file())
    lines = ["%s  %s" % (sha256(p), p.relative_to(out).as_posix()) for p in files]
    (out / "CODE_MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    meta = {
        "title": "Solar filament segmentation: code, folds, weights",
        "id": "%s/%s" % (OWNER, DATASET_SLUG),
        "licenses": [{"name": "other"}],
        "description": "Private code snapshot (code.zip, commit %s), folds.csv with its provenance, optional ImageNet weights. "
                       "Contains competition file names: keep private." % gi["git_commit"][:10],
    }
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
    total = sum(p.stat().st_size for p in files)
    print("dataset folder: %s" % out)
    print("files: %d, %.1f MB, commit %s, weights: %s" % (len(files), total / 1e6, gi["git_commit"][:10], weights))
    print("next (with the user's Kaggle credentials): kaggle datasets create -p %s   (first time)" % out)
    print("                                           kaggle datasets version -p %s -m \"<message>\"   (a later version)" % out)


if __name__ == "__main__":
    main()
