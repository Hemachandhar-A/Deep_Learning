"""Print what the first Kaggle session has to find out (development prompt section 9, Arch section 4.8).

    python kaggle/probe_environment.py [--folds-csv <path>] [--two-gpu] [--out probe_report.json]

Everything printed is read from the running session; nothing is assumed.  The unknowns it addresses:

* the path of the competition data (searched for by the annotation file name; verified on Kaggle on 2026-10-14:
  ``/kaggle/input/competitions/<slug>/...``, and datasets under ``/kaggle/input/datasets/<owner>/<slug>/``);
* whether the Internet is reachable with the competition attached (Hugging Face Hub and PyPI, HEAD requests);
* the GPUs, their memory, torch / CUDA / package versions, free disk and RAM;
* persistence of ``/kaggle/working`` between sessions: it writes a marker file and reports any marker from an earlier
  session that was attached as an input;
* whether two training processes on two cards run side by side at full speed (``--two-gpu``: one benchmark alone, then
  one per card at the same time);
* the SHA-256 of ``folds.csv`` and the scikit-learn version, if ``--folds-csv`` is given;
* the run record of Phase 2.1: CPU count, GPU count and names, the library versions, the approved step budget
  (``--step-budget``), where the cache goes (``--cache-dir``) and whether ``/kaggle/temp`` is usable (``--temp-root``).

Not readable from inside a notebook: the weekly GPU quota.  It is shown in the Kaggle web page (profile -> Settings ->
Notebook usage / GPU quota); write the number into the report.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
JSON_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def find_data(roots=("/kaggle/input",)) -> list[str]:
    hits = []
    for r in roots:
        root = Path(r)
        if root.exists():
            hits += [str(p) for p in root.rglob(JSON_NAME)]
    return hits


def list_inputs(root="/kaggle/input", depth=2) -> list[str]:
    out = []
    base = Path(root)
    if not base.exists():
        return out
    for p in sorted(base.rglob("*")):
        if len(p.relative_to(base).parts) <= depth and p.is_dir():
            out.append(str(p))
    return out[:200]


def reach(url: str, timeout: float = 15.0) -> dict:
    t0 = time.time()
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {"url": url, "reachable": True, "status": r.status, "seconds": round(time.time() - t0, 2)}
    except Exception as e:  # any failure means "not reachable from here"
        return {"url": url, "reachable": False, "error": "%s: %s" % (type(e).__name__, str(e)[:120])}


def run_benchmark(env_extra: dict, steps: int = 20) -> dict:
    env = {**os.environ, **env_extra}
    p = subprocess.run([sys.executable, str(HERE / "probe_step_benchmark.py"), "--steps", str(steps)],
                       capture_output=True, text=True, env=env, timeout=900)
    line = (p.stdout.strip().splitlines() or [""])[-1]
    try:
        return json.loads(line)
    except Exception:
        return {"error": (p.stderr or p.stdout)[-300:]}


def two_gpu_test(steps: int = 20) -> dict:
    import torch

    n = torch.cuda.device_count()
    res = {"gpus_visible": n}
    if n < 2:
        res["note"] = "fewer than two GPUs visible: the side-by-side test needs two"
        return res
    res["alone_gpu0"] = run_benchmark({"CUDA_VISIBLE_DEVICES": "0"}, steps)
    procs = []
    for g in (0, 1):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
        procs.append(subprocess.Popen([sys.executable, str(HERE / "probe_step_benchmark.py"), "--steps", str(steps)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env))
    both = []
    for p in procs:
        out, err = p.communicate(timeout=900)
        try:
            both.append(json.loads(out.strip().splitlines()[-1]))
        except Exception:
            both.append({"error": (err or out)[-300:]})
    res["concurrent"] = both
    try:
        a = res["alone_gpu0"]["seconds_per_step"]
        res["slowdown_vs_alone"] = [round(b["seconds_per_step"] / a, 3) for b in both]
    except Exception:
        pass
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds-csv", type=Path, default=None)
    ap.add_argument("--two-gpu", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--working", type=Path, default=Path("/kaggle/working"))
    ap.add_argument("--input-root", type=Path, default=Path("/kaggle/input"), help="where Kaggle mounts inputs")
    ap.add_argument("--step-budget", type=int, default=None, help="the approved train.steps, for the run record")
    ap.add_argument("--cache-dir", type=Path, default=None, help="the cache folder the notebook chose")
    ap.add_argument("--temp-root", type=Path, default=Path("/kaggle/temp"), help="candidate folder outside the output")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rep: dict = {"time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}

    rep["python"] = platform.python_version()
    rep["cpu_count"] = os.cpu_count()
    rep["platform"] = platform.platform()
    try:
        import torch

        rep["torch"] = torch.__version__
        rep["cuda_available"] = torch.cuda.is_available()
        rep["gpus"] = [{"name": torch.cuda.get_device_name(i),
                        "total_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2)}
                       for i in range(torch.cuda.device_count())]
        rep["gpu_count"] = len(rep["gpus"])
    except Exception as e:
        rep["torch_error"] = str(e)
    import importlib.metadata as md

    rep["packages"] = {}
    for p in ["segmentation_models_pytorch", "timm", "numpy", "scipy", "opencv-python", "opencv-python-headless",
              "scikit-learn", "pycocotools", "pillow", "PyYAML", "huggingface_hub", "pandas", "nbformat"]:
        try:
            rep["packages"][p] = md.version(p)
        except md.PackageNotFoundError:
            rep["packages"][p] = None
    du = shutil.disk_usage(str(a.working if a.working.exists() else Path(".")))
    rep["disk_free_gib_working"] = round(du.free / 2**30, 1)
    try:
        import psutil

        rep["ram_total_gib"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        rep["ram_total_gib"] = None

    rep["step_budget"] = a.step_budget
    from filament import runinfo

    rep["cache_location"] = runinfo.cache_location_label(a.cache_dir)
    t = a.temp_root  # not assumed to exist: reported as found
    rep["temp_folder"] = {"path": t.as_posix(), "exists": t.is_dir()}
    if t.is_dir():
        try:
            probe_file = t / ".probe_write_test"
            probe_file.write_bytes(b"x")
            probe_file.unlink()
            rep["temp_folder"]["writable"] = True
        except OSError as e:
            rep["temp_folder"]["writable"] = False
            rep["temp_folder"]["error"] = str(e)
        rep["temp_folder"]["free_gib"] = round(shutil.disk_usage(str(t)).free / 2**30, 1)
    rep["kaggle_input_dirs"] = list_inputs(str(a.input_root), depth=4)  # a dataset may be mounted a few levels deep
    rep["data_json_locations"] = find_data((str(a.input_root),))
    rep["internet"] = [reach("https://huggingface.co/smp-hub/resnet34.imagenet/resolve/main/config.json"),
                       reach("https://pypi.org/simple/segmentation-models-pytorch/")]

    # persistence between sessions: leave a marker; report markers of earlier sessions attached as inputs
    marker = a.working / "probe_marker.json"
    rep["earlier_markers_in_inputs"] = [str(p) for p in a.input_root.rglob("probe_marker.json")] \
        if a.input_root.exists() else []
    try:
        a.working.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"written_utc": rep["time_utc"]}), encoding="utf-8")
        rep["working_writable"] = True
    except Exception as e:
        rep["working_writable"] = False
        rep["working_error"] = str(e)

    if a.folds_csv:
        h = hashlib.sha256(Path(a.folds_csv).read_bytes()).hexdigest()
        import sklearn

        rep["folds_csv_sha256"] = h
        rep["scikit_learn_version"] = sklearn.__version__
        side = Path(str(a.folds_csv) + ".provenance.json")
        if side.exists():
            exp = json.loads(side.read_text(encoding="utf-8")).get("folds_csv_sha256")
            rep["folds_matches_sidecar"] = bool(exp) and exp == h
    if a.two_gpu:
        rep["two_gpu_side_by_side"] = two_gpu_test(a.steps)
    rep["not_readable_from_here"] = ["weekly GPU quota (read it in the Kaggle web page)"]

    text = json.dumps(rep, indent=1)
    print(text)
    if a.out:
        a.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
