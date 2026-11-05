"""The Kaggle package (kaggle/) and the pipeline notebook (notebooks/): safe defaults, metadata, build == source,
dataset contents.  No Kaggle access, no token, nothing is uploaded by any of this."""
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
K = REPO / "kaggle"
sys.path.insert(0, str(K))

import build_notebook  # noqa: E402
import prepare_push_folder  # noqa: E402

# shapes of real secrets (a Kaggle API token, a Hugging Face token, a kaggle.json key/username pair), not the words
SECRET = re.compile(r"(KGAT_[A-Za-z0-9]{8,}|hf_[A-Za-z0-9]{20,}|\"key\"\s*:\s*\"[0-9a-f]{16,}\"|\"username\"\s*:\s*\"[^\"]+\"\s*,\s*\"key\")")


def test_notebook_defaults_cannot_train_by_accident():
    src = (K / "notebook_source.py").read_text(encoding="utf-8")
    assert re.search(r'^MODE = "probe"', src, re.M) and re.search(r"^STEPS = None", src, re.M)
    assert re.search(r"^DRY = False", src, re.M) and re.search(r"^LRCHECK_ROWS = None", src, re.M)
    assert re.search(r"^INFER_AFTER = False", src, re.M)  # the inference after training is a switch, off by default
    assert "OFFLINE_WEIGHTS = False" in src and "RUN_ID_FOR_INFER = None" in src
    # the step budget comes from the config; another number is refused unless it is a pipeline check; null refuses
    assert "assert DRY," in src and "no step budget" in src and "def effective_steps" in src
    assert src.count("effective_steps(") >= 3  # e1/e1b and lrcheck both go through it
    assert "competitions submit" not in src  # the notebook never calls the submission API


def test_one_place_for_the_step_budget_and_every_training_config_inherits_it():
    """Reviewer's budget (2026-10-15): 7,000 steps per fold, written once, read by every training mode."""
    import sys as _sys

    _sys.path.insert(0, str(REPO / "src"))
    from filament.config import load_config

    base = load_config(REPO / "configs" / "base.yaml")
    assert base["train"]["steps"] == 7000
    for name in ("e1_seed1.yaml", "e1b_seed2.yaml", "e1_lrcheck.yaml"):
        assert load_config(REPO / "configs" / name)["train"]["steps"] == 7000, name
        text = (REPO / "configs" / name).read_text(encoding="utf-8")
        assert "steps:" not in text.replace("train.steps", ""), "%s sets steps itself: one place only" % name
    assert prepare_push_folder.approved_budget() == 7000


def test_kernel_metadata_is_explicit_and_belongs_to_the_user():
    meta = json.loads((K / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert prepare_push_folder.check_metadata(meta) == []
    assert meta["enable_gpu"] is True and meta["enable_internet"] is True and meta["is_private"] is True
    assert meta["competition_sources"] == ["filament-segmentation-2026"] and meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["dataset_sources"] == ["hemachandhara/solar-filament-code"]
    bad = dict(meta)
    bad.pop("enable_gpu")
    bad["enable_internet"] = "yes"
    problems = prepare_push_folder.check_metadata(bad)
    assert any("missing" in p for p in problems) and any("enable_internet" in p for p in problems)


def test_title_slug_equals_id_slug_and_the_metadata_bytes_are_the_reviewed_ones():
    """Phase 2.1: Kaggle names a kernel after its title; a mismatch made the second push fail with HTTP 409."""
    meta = json.loads((K / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert prepare_push_folder.slug(meta["title"]) == meta["id"].split("/")[-1] == "solar-filament-segmentation-pipeline"
    assert meta["id"] == "hemachandhara/solar-filament-segmentation-pipeline"
    bad = {**meta, "id": "hemachandhara/solar-filament-pipeline"}  # the id of the 409 on 2026-10-15
    assert any("slug" in p and "409" in p for p in prepare_push_folder.check_metadata(bad))
    assert prepare_push_folder.slug("  Solar filament -- segmentation pipeline! ") == "solar-filament-segmentation-pipeline"
    # the reviewer's reference for the push folder's metadata file
    sha = hashlib.sha256((K / "kernel-metadata.json").read_bytes()).hexdigest()
    assert sha == "e23105cef5565ee8c97a179bb010413d313e6394ddb863a4c62f0ffef9ff0235"


def test_no_secret_in_the_kaggle_or_notebook_folders():
    for folder in (K, REPO / "notebooks"):
        for f in folder.rglob("*"):
            if f.is_file() and f.suffix in {".py", ".json", ".md", ".ipynb"}:
                assert not SECRET.search(f.read_text(encoding="utf-8")), f.name


@pytest.mark.parametrize("source", ["kaggle/notebook_source.py", "notebooks/pipeline_source.py"])
def test_built_notebook_equals_its_source_and_compiles(source, tmp_path):
    out = tmp_path / "nb.ipynb"
    n = build_notebook.build(REPO / source, out)
    nb = json.loads(out.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and len(nb["cells"]) == n >= 6
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code and all(c["outputs"] == [] and c["execution_count"] is None for c in code)
    for i, c in enumerate(code):
        compile("".join(c["source"]), "%s cell %d" % (source, i), "exec")
    # the concatenated code cells are exactly the code lines of the source (nothing added, nothing edited by hand)
    src_code, in_md = [], False
    for line in (REPO / source).read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            in_md = "[markdown]" in line
            continue
        if not in_md:
            src_code.append(line)
    built = "\n".join("".join(c["source"]) for c in code)
    norm = lambda t: [l for l in t.splitlines() if l.strip()]  # noqa: E731
    assert norm(built) == norm("\n".join(src_code))


@pytest.mark.parametrize("source, built", [("notebooks/pipeline_source.py", "notebooks/pipeline.ipynb"),
                                            ("kaggle/notebook_source.py", "kaggle/kaggle_notebook.ipynb")])
def test_committed_notebooks_are_the_build_of_their_sources(source, built, tmp_path):
    committed = REPO / built
    assert committed.exists(), "%s is not built: python kaggle/build_notebook.py %s %s" % (built, source, built)
    tmp = tmp_path / "check.ipynb"
    build_notebook.build(REPO / source, tmp)
    assert committed.read_text(encoding="utf-8") == tmp.read_text(encoding="utf-8")


def test_push_folder_contains_only_notebook_and_metadata(tmp_path):
    out = tmp_path / "push"
    p = subprocess.run([sys.executable, str(K / "prepare_push_folder.py"), "--out", str(out)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert sorted(f.name for f in out.iterdir()) == ["kaggle_notebook.ipynb", "kernel-metadata.json"]
    again = subprocess.run([sys.executable, str(K / "prepare_push_folder.py"), "--out", str(out)], capture_output=True, text=True)
    assert again.returncode != 0  # never writes into a non-empty folder


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_dataset_holds_tracked_files_only_and_a_checkable_manifest(tmp_path):
    from filament import folds, io_utils

    v = {"2015%010d%s" % (i, ("Bh", "Ch", "Lh", "Mh", "Th", "Uh")[i % 6]): [0] * (1 + i % 3) for i in range(120)}
    fp = tmp_path / "folds.csv"
    folds.write_folds_csv(fp, folds.make_folds(v))
    io_utils.atomic_write_json(folds.sidecar_path(fp), {**folds.provenance(fp), "strategy": "balanced", "seed": 42})
    out = tmp_path / "dataset"
    p = subprocess.run([sys.executable, str(K / "make_dataset.py"), "--folds-csv", str(fp), "--out", str(out),
                        "--allow-dirty"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    import zipfile
    top = sorted(f.name for f in out.iterdir())
    assert top == ["CODE_MANIFEST.sha256", "COMMIT", "code.zip", "dataset-metadata.json", "folds.csv",
                   "folds.csv.provenance.json"], top  # flat: kaggle datasets create skips sub-folders
    tracked = set(subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True).stdout.split())
    with zipfile.ZipFile(out / "code.zip") as z:
        names = {n[len("repo/"):] for n in z.namelist() if n.startswith("repo/") and not n.endswith("/")}
        inner = z.read("repo/REPO_MANIFEST.sha256").decode("utf-8").splitlines()
        assert z.testzip() is None
        assert names - {"COMMIT", "REPO_MANIFEST.sha256"} <= tracked, sorted(names - tracked)
        assert not [n for n in names if n.endswith((".npy", ".pt", ".pth", ".jpeg", ".jpg", ".safetensors"))]
        assert not [n for n in names if n.startswith("configs/local")]
        for line in inner:  # the inner manifest checks every file of the tree
            h, rel = line.split("  ", 1)
            assert hashlib.sha256(z.read("repo/" + rel)).hexdigest() == h, rel
        assert len(inner) > 30 and "src/filament/train.py" in {l.split("  ", 1)[1] for l in inner}
        assert z.read("repo/COMMIT").decode().strip() == (out / "COMMIT").read_text().strip()
    # the top-level manifest covers code.zip and the folds
    lines = (out / "CODE_MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    for line in lines:
        h, rel = line.split("  ", 1)
        assert hashlib.sha256((out / rel).read_bytes()).hexdigest() == h
    assert {l.split("  ", 1)[1] for l in lines} == set(top) - {"CODE_MANIFEST.sha256", "dataset-metadata.json"}
    assert (out / "folds.csv").read_bytes() == fp.read_bytes()
    assert (out / "folds.csv.provenance.json").exists() and (out / "COMMIT").read_text().strip()
    meta = json.loads((out / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "hemachandhara/solar-filament-code" and meta["licenses"] == [{"name": "other"}]
    # a folds file that does not match its sidecar is refused
    fp.write_bytes(fp.read_bytes() + b"\n")
    bad = subprocess.run([sys.executable, str(K / "make_dataset.py"), "--folds-csv", str(fp), "--out", str(tmp_path / "d2"),
                          "--allow-dirty"], capture_output=True, text=True)
    assert bad.returncode != 0 and "SHA-256" in (bad.stdout + bad.stderr)


def _push(tmp_path, *args):
    out = tmp_path / "push"
    p = subprocess.run([sys.executable, str(K / "prepare_push_folder.py"), "--out", str(out), *args],
                       capture_output=True, text=True)
    return p, out


def test_push_folder_sets_only_the_parameter_lines(tmp_path):
    p, out = _push(tmp_path, "--mode", "measure")
    assert p.returncode == 0, p.stderr
    built = json.loads((out / "kaggle_notebook.ipynb").read_text(encoding="utf-8"))
    text = "".join("".join(c["source"]) + "\n" for c in built["cells"] if c["cell_type"] == "code")
    assert 'MODE = "measure"' in text and "STEPS = None" in text
    ref = json.loads((K / "kaggle_notebook.ipynb").read_text(encoding="utf-8"))
    ref_text = "".join("".join(c["source"]) + "\n" for c in ref["cells"] if c["cell_type"] == "code")
    diff = [(a, b) for a, b in zip(ref_text.splitlines(), text.splitlines()) if a != b]
    assert diff == [('MODE = "probe"  # "probe" | "measure" | "lrcheck" | "e1" | "e1b" | "exp" | "infer"',
                     'MODE = "measure"  # "probe" | "measure" | "lrcheck" | "e1" | "e1b" | "exp" | "infer"')]


def test_training_modes_use_the_budget_of_the_config_and_refuse_another_number(tmp_path):
    for mode in ("e1", "e1b", "lrcheck"):  # no --steps: the notebook reads 7,000 from the config
        p, out = _push(tmp_path / mode, "--mode", mode)
        assert p.returncode == 0, p.stderr
        assert "7000 per fold" in p.stdout
        text = (out / "kaggle_notebook.ipynb").read_text(encoding="utf-8")
        assert "STEPS = None" in text and "DRY = False" in text and 'MODE = \\"%s\\"' % mode in text
    p, _ = _push(tmp_path / "other", "--mode", "e1", "--steps", "4000")  # a different number without --dry: refused
    assert p.returncode != 0 and "approved budget 7000" in (p.stdout + p.stderr)
    p, _ = _push(tmp_path / "same", "--mode", "e1", "--steps", "7000")  # the budget itself is fine
    assert p.returncode == 0, p.stderr
    p, out = _push(tmp_path / "dry", "--mode", "e1", "--steps", "6", "--dry", "--folds", "0")
    assert p.returncode == 0 and "OVERRIDE" in p.stdout
    text = (out / "kaggle_notebook.ipynb").read_text(encoding="utf-8")
    assert "STEPS = 6" in text and "DRY = True" in text and "FOLDS = [0]" in text
    bad, _ = _push(tmp_path / "neg", "--mode", "e1", "--steps", "0", "--dry")
    assert bad.returncode != 0
    noid, _ = _push(tmp_path / "infer", "--mode", "infer")
    assert noid.returncode != 0 and "--run-id" in (noid.stdout + noid.stderr)
    rows, out = _push(tmp_path / "rows", "--mode", "lrcheck", "--steps", "6", "--dry", "--lr-rows", "1", "3")
    assert rows.returncode == 0 and "LRCHECK_ROWS = [1, 3]" in (out / "kaggle_notebook.ipynb").read_text(encoding="utf-8")
    wrong, _ = _push(tmp_path / "rows2", "--mode", "e1", "--lr-rows", "1")
    assert wrong.returncode != 0
    ia, out = _push(tmp_path / "ia", "--mode", "e1", "--infer-after")  # Phase 2.2: only e1 and e1b, budget from the config
    assert ia.returncode == 0 and "INFER_AFTER = True" in (out / "kaggle_notebook.ipynb").read_text(encoding="utf-8")
    assert _push(tmp_path / "ia_lr", "--mode", "lrcheck", "--infer-after")[0].returncode != 0
    assert _push(tmp_path / "ia_probe", "--mode", "probe", "--infer-after")[0].returncode != 0
    # an unset budget refuses every training mode (A4.3), at build time too
    assert prepare_push_folder.check_budget("e1", None, False, None)
    assert prepare_push_folder.check_budget("lrcheck", None, False, None)
    assert prepare_push_folder.check_budget("probe", None, False, None) == []  # probe and measure never need it
    assert prepare_push_folder.check_budget("measure", None, False, None) == []
    assert prepare_push_folder.check_budget("e1", 6, True, 7000) == []
    assert prepare_push_folder.check_budget("probe", 6, True, 7000)  # --dry is not for probe


def _notebook_cells(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _run_cells(cells, input_dir, work_dir):
    """Execute the parameter cell and the code-dataset cell of the Kaggle notebook against temporary folders."""
    params = next(c for c in cells if "INPUT = Path(" in c)
    find = next(c for c in cells if "REPO_MANIFEST.sha256" in c)
    params = params.replace('WORK = Path("/kaggle/working")', "WORK = Path(%r)" % str(work_dir))
    params = params.replace('INPUT = Path("/kaggle/input")', "INPUT = Path(%r)" % str(input_dir))
    ns = {}
    exec(compile(params, "notebook parameter cell", "exec"), ns)
    exec(compile(find, "notebook code-dataset cell", "exec"), ns)
    return ns


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_notebook_finds_the_code_dataset_at_any_depth_and_lists_folders_when_it_is_absent(tmp_path, monkeypatch, capsys):
    from filament import folds, io_utils

    monkeypatch.setattr(sys, "path", list(sys.path))  # the cell puts the unpacked code on sys.path
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "0")  # the first cell sets it to 1
    v = {"2015%010d%s" % (i, ("Bh", "Ch", "Lh", "Mh", "Th", "Uh")[i % 6]): [0] * (1 + i % 3) for i in range(120)}
    fp = tmp_path / "folds.csv"
    folds.write_folds_csv(fp, folds.make_folds(v))
    io_utils.atomic_write_json(folds.sidecar_path(fp), {**folds.provenance(fp), "strategy": "balanced", "seed": 42})
    built = tmp_path / "built"
    p = subprocess.run([sys.executable, str(K / "make_dataset.py"), "--folds-csv", str(fp), "--out", str(built),
                        "--allow-dirty"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    cells = _notebook_cells(K / "kaggle_notebook.ipynb")
    # mounted three folders deep (one possible layout of /kaggle/input): found, manifests verified, code unpacked
    nested = tmp_path / "input" / "datasets" / "hemachandhara" / "solar-filament-code"
    shutil.copytree(built, nested)
    ns = _run_cells(cells, tmp_path / "input", tmp_path / "working")
    assert ns["CODE"] == nested and (ns["REPO"] / "src" / "filament" / "train.py").exists()
    assert ns["REPO"] == tmp_path / "working" / "code" / "repo"
    # a damaged code.zip is refused before anything from it runs
    (nested / "code.zip").write_bytes((nested / "code.zip").read_bytes() + b"x")
    with pytest.raises(AssertionError, match="differ from CODE_MANIFEST"):
        _run_cells(cells, tmp_path / "input", tmp_path / "working2")
    # nothing attached: the message names the problem and the folders that exist are printed first
    empty = tmp_path / "input2"
    (empty / "some-competition" / "train").mkdir(parents=True)
    capsys.readouterr()
    with pytest.raises(AssertionError, match="no CODE_MANIFEST.sha256"):
        _run_cells(cells, empty, tmp_path / "working3")
    assert "some-competition" in capsys.readouterr().out


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_notebook_glue_on_kaggles_unpacked_layout_offline_weights_and_budget_rules(tmp_path, monkeypatch):
    """Phase 2.1: Kaggle unpacks both zips (code/repo/..., hf_cache/hf_cache/hub/...) and mounts the competition under
    /kaggle/input/competitions/<slug>/.  Runs the parameter, code-dataset, weights and helper cells on that layout."""
    import os
    import zipfile

    from filament import data, folds, io_utils, oof, procs, runinfo, train
    from filament.config import load_config

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "0")
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    v = {"2015%010d%s" % (i, ("Bh", "Ch", "Lh", "Mh", "Th", "Uh")[i % 6]): [0] * (1 + i % 3) for i in range(120)}
    fp = tmp_path / "folds.csv"
    folds.write_folds_csv(fp, folds.make_folds(v))
    io_utils.atomic_write_json(folds.sidecar_path(fp), {**folds.provenance(fp), "strategy": "balanced", "seed": 42})
    built = tmp_path / "built"
    p = subprocess.run([sys.executable, str(K / "make_dataset.py"), "--folds-csv", str(fp), "--out", str(built),
                        "--allow-dirty"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    # the layout of the probe of 2026-10-14: the zips are unpacked by Kaggle, the files keep the manifest names
    mount = tmp_path / "input" / "datasets" / "hemachandhara" / "solar-filament-code"
    mount.mkdir(parents=True)
    for name in ("CODE_MANIFEST.sha256", "COMMIT", "folds.csv", "folds.csv.provenance.json", "dataset-metadata.json"):
        shutil.copy2(built / name, mount / name)
    with zipfile.ZipFile(built / "code.zip") as z:
        z.extractall(mount / "code")
    model = mount / "hf_cache" / "hf_cache" / "hub" / "models--smp-hub--resnet34.imagenet" / "snapshots" / "rev"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    comp = tmp_path / "input" / "competitions" / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026"
    (comp / "train" / "train_images").mkdir(parents=True)
    (comp / "test" / "test_images").mkdir(parents=True)
    (comp / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_text("{}", encoding="utf-8")

    cells = _notebook_cells(K / "kaggle_notebook.ipynb")
    params = next(c for c in cells if "INPUT = Path(" in c)
    for old, new in (('WORK = Path("/kaggle/working")', "WORK = Path(%r)" % str(tmp_path / "working")),
                     ('TEMP = Path("/kaggle/temp")', "TEMP = Path(%r)" % str(tmp_path / "temp")),
                     ('INPUT = Path("/kaggle/input")', "INPUT = Path(%r)" % str(tmp_path / "input")),
                     ("OFFLINE_WEIGHTS = False", "OFFLINE_WEIGHTS = True")):
        assert params.count(old) == 1, old
        params = params.replace(old, new)
    ns = {}
    exec(compile(params, "parameter cell", "exec"), ns)
    exec(compile(next(c for c in cells if "REPO_MANIFEST.sha256" in c), "code-dataset cell", "exec"), ns)
    assert ns["CODE"] == mount and ns["REPO"] == mount / "code" / "repo"
    assert (ns["REPO"] / "src" / "filament" / "train.py").exists()

    # offline weights: the hub folder is searched for, not assumed (G2-13); the competition is found where Kaggle put it
    ns.update(data=data, folds=folds, io_utils=io_utils, oof=oof, procs=procs, runinfo=runinfo, train=train,
              load_config=load_config)
    exec(compile(next(c for c in cells if "if OFFLINE_WEIGHTS:" in c), "weights cell", "exec"), ns)
    assert Path(os.environ["HF_HUB_CACHE"]) == mount / "hf_cache" / "hf_cache" / "hub"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert ns["layout"].train_json.parent == comp / "train" and ns["layout"].test_images == comp / "test" / "test_images"

    # the budget rules and the command lines of the training modes
    ns.update(BUDGET=7000, CACHE=tmp_path / "temp" / "cache", CACHE_WHY="test", FOLDS_CSV=tmp_path / "working" / "folds.csv")
    exec(compile(next(c for c in cells if "def effective_steps" in c), "helper cell", "exec"), ns)
    assert ns["effective_steps"]("e1_seed1.yaml") == 7000 and ns["effective_steps"]("e1_lrcheck.yaml") == 7000
    ns["STEPS"] = 4000
    with pytest.raises(AssertionError, match="approved budget 7000"):
        ns["effective_steps"]("e1_seed1.yaml")
    ns["DRY"] = True
    assert ns["effective_steps"]("e1_seed1.yaml") == 4000  # a pipeline check may use another number
    ns["STEPS"], ns["DRY"] = None, False
    assert ns["run_id_for"]("e1_lrcheck.yaml", lr=3e-4, seed=1) == "E1lr_r512_lr3e-4_seed1"
    assert ns["run_id_for"]("e1b_seed2.yaml") == "E1b_r512_lr1e-3_seed2"
    assert ns["run_id_for"]("e1_lrcheck.yaml", lr=1e-4, seed=2) == "E1lr_r512_lr1e-4_seed2"
    assert ns["run_id_for"]("e1c_seed3.yaml") == "E1c_r512_lr1e-3_seed3"  # Phase 3a arms: the run id comes from the config
    assert ns["run_id_for"]("e6b_flip_rot90_brightness.yaml") == "E6b_r512_lr1e-3_seed1"
    assert ns["effective_steps"]("e4b_tversky.yaml") == 7000 and ns["effective_steps"]("e6a_flip_rot90.yaml") == 7000
    cmd = ns["train_cmd"]("e1_lrcheck.yaml", 0, lr=3e-4, seed=2, steps=7000)
    assert cmd[cmd.index("--fold") + 1] == "0" and cmd[cmd.index("--lr") + 1] == "0.0003"
    assert cmd[cmd.index("--seed") + 1] == "2" and cmd[cmd.index("--steps") + 1] == "7000" and "--mark-dry" not in cmd
    ns["DRY"] = True
    assert ns["run_id_for"]("e1_seed1.yaml") == "DRY_E1_r512_lr1e-3_seed1"
    assert "--mark-dry" in ns["train_cmd"]("e1_seed1.yaml", 1, steps=6)
    # an unset budget in the code that was uploaded refuses every training mode
    base = ns["REPO"] / "configs" / "base.yaml"
    base.write_text(base.read_text(encoding="utf-8").replace("steps: 7000", "steps: null", 1), encoding="utf-8")
    ns["STEPS"], ns["DRY"] = None, False
    with pytest.raises(AssertionError, match="no step budget"):
        ns["effective_steps"]("e1_seed1.yaml")


def test_finish_step_survives_a_scoring_error_runs_inference_in_order_and_raises_at_the_very_end(tmp_path, monkeypatch, capsys):
    """Phase 2.2: F3 (any scoring error is caught, the manifest and the listing are still written) and option A (inference
    after training, in the order of the Gate 2b review: aggregation, scores, manifest, listing, inference, second
    manifest, raise).  The notebook's own helper cell runs against a real tiny CPU run; scoring and the model-based
    inference are replaced by stubs so that the glue is what is tested."""
    import types

    import numpy as np

    from filament import data, infer, io_utils, oof, procs, runinfo, train
    from filament.config import load_config
    from synthetic import DRY_INFO, make_cache, make_cfg, rows_for

    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "0")
    cells = _notebook_cells(K / "kaggle_notebook.ipynb")
    work = tmp_path / "working"
    params = next(c for c in cells if "INPUT = Path(" in c)
    for old, new in (('WORK = Path("/kaggle/working")', "WORK = Path(%r)" % str(work)),
                     ('TEMP = Path("/kaggle/temp")', "TEMP = Path(%r)" % str(tmp_path / "temp")),
                     ('INPUT = Path("/kaggle/input")', "INPUT = Path(%r)" % str(tmp_path / "input"))):
        params = params.replace(old, new)
    ns = {}
    exec(compile(params, "parameter cell", "exec"), ns)
    cache = make_cache()
    m = train.run_training(make_cfg(), cache, rows_for(cache), DRY_INFO, [0], work, work / "results", "cpu", dry_run=True,
                           log=lambda *_: None)
    rid = m["run_id"]
    calls = []

    def fake_run_inference(cfg, cache_, run_dir, out_dir, device, folds=None, flips=("",)):
        calls.append(("inference", folds))
        io_utils.atomic_write_text(out_dir / "submission.csv", "id,rle\n")
        io_utils.atomic_save_npy(out_dir / "test_probs.npy", np.zeros((2, 4, 4), np.float16))
        return {"csv_rows": 0, "images": 2, "csv_check_ok": True, "round_trip_mismatches": 0, "components_per_image": {},
                "csv_sha256": "x", "folds_used": folds}

    score_row = {"params": {"threshold": 0.5}, "pq": 0.1, "pq_ci_lo": 0.0, "pq_ci_hi": 0.2, "SQ": 0.5, "RQ": 0.2, "TP": 1,
                 "FP": 2, "FN": 3, "pred_per_image_mean": 4.0}

    def boom(*a, **k):
        raise RuntimeError("simulated out of memory")

    ns.update(
        REPO=REPO, BUDGET=7000, CACHE=tmp_path / "temp" / "cache", CACHE_WHY="test", FOLDS_CSV=work / "folds.csv",
        layout=types.SimpleNamespace(train_json="unused"), torch=__import__("torch"), load_config=load_config,
        data=types.SimpleNamespace(load_annotations=lambda p: object(), load_cache=lambda c: cache),
        io_utils=io_utils, procs=procs, runinfo=runinfo, train=train,
        infer_mod=types.SimpleNamespace(plan_inference_after_training=infer.plan_inference_after_training,
                                        write_infer_manifest=infer.write_infer_manifest, run_inference=fake_run_inference),
        oof=types.SimpleNamespace(score_run=boom),
        DRY=True, FOLDS=[0], STEPS=None)
    exec(compile(next(c for c in cells if "def finish_training_session" in c), "helper cell", "exec"), ns)
    finish = ns["finish_training_session"]
    ok = [("e1 fold 0", 0)]

    # F3: the scoring raises something other than FileNotFoundError -> traceback printed, manifest and listing still written
    with pytest.raises(procs.ProcessFailures) as e:
        finish(ok, "e1", [rid])
    cap = capsys.readouterr()
    assert "scoring of %s failed" % rid in str(e.value) and "processes failed" not in str(e.value)
    assert "RuntimeError: simulated out of memory" in cap.err and "output folder:" in cap.out
    assert (work / "results" / rid / "MANIFEST.sha256").exists() and not (work / "results" / rid / "oof_scores.json").exists()
    assert json.loads((work / "results" / rid / "metrics.json").read_text(encoding="utf-8"))["folds_trained"] == [0]

    # scoring works, inference requested in a DRY session with the one fold that trained: runs, in the specified order
    ns["oof"] = types.SimpleNamespace(score_run=lambda *a, **k: {"results": [score_row]})
    finish(ok, "e1", [rid], infer=("e1_seed1.yaml", rid))
    out = capsys.readouterr().out
    order = [out.index(s) for s in ("manifest: results/", "output folder:", "inference after training: RUNNING", "new files:")]
    assert order == sorted(order) and out.rindex("manifest: results/") > order[1]  # second manifest after the inference
    assert calls == [("inference", [0])]
    assert (work / "DRY_submission_candidate" / "submission.csv").exists()
    lines = (work / "results" / rid / "MANIFEST_infer.sha256").read_text(encoding="utf-8").splitlines()
    assert [l.split("  ", 1)[1] for l in lines] == ["DRY_submission_candidate/submission.csv", "DRY_submission_candidate/test_probs.npy"]
    assert (work / "results" / rid / "oof_scores.json").exists()

    # an error inside the inference: printed, nothing removed, the session still ends in an error naming it
    ns["infer_mod"].run_inference = boom
    with pytest.raises(procs.ProcessFailures, match="inference after training failed"):
        finish(ok, "e1", [rid], infer=("e1_seed1.yaml", rid))
    cap = capsys.readouterr()
    assert "RuntimeError: simulated out of memory" in cap.err
    assert (work / "results" / rid / "MANIFEST.sha256").exists() and (work / "DRY_submission_candidate" / "submission.csv").exists()

    # a real session of only some folds trains but does not infer, and is not an error by itself
    ns["infer_mod"].run_inference = fake_run_inference
    ns["DRY"] = False
    calls.clear()
    finish(ok, "e1", [rid], infer=("e1_seed1.yaml", rid))
    assert "not run" in capsys.readouterr().out and calls == []
    # a failed process: no inference, and the error names the process
    with pytest.raises(procs.ProcessFailures, match=r"e1 fold 1 \(exit code 1\)"):
        finish(ok + [("e1 fold 1", 1)], "e1", [rid], infer=("e1_seed1.yaml", rid))
    assert calls == [] and "not run: 1 process(es) failed" in capsys.readouterr().out


def test_exp_mode_push_folder_sets_only_its_parameter_lines_and_refuses_misuse(tmp_path):
    """Phase 3a: --mode exp --config <name>.yaml; the budget is the config's own; --config belongs to exp only."""
    p, out = _push(tmp_path / "ok", "--mode", "exp", "--config", "e1c_seed3.yaml")
    assert p.returncode == 0, p.stderr
    assert "7000 per fold" in p.stdout and "CONFIG='e1c_seed3.yaml'" in p.stdout
    text = (out / "kaggle_notebook.ipynb").read_text(encoding="utf-8")
    assert 'MODE = \\"exp\\"' in text and 'CONFIG = \\"e1c_seed3.yaml\\"' in text and "STEPS = None" in text and "DRY = False" in text
    # only the parameter lines differ from the committed notebook (MODE and CONFIG here)
    built = json.loads(text)
    got = "".join("".join(c["source"]) + "\n" for c in built["cells"] if c["cell_type"] == "code").splitlines()
    ref = json.loads((K / "kaggle_notebook.ipynb").read_text(encoding="utf-8"))
    want = "".join("".join(c["source"]) + "\n" for c in ref["cells"] if c["cell_type"] == "code").splitlines()
    assert len(got) == len(want)
    diff = [(a.split("  #")[0], b.split("  #")[0]) for a, b in zip(want, got) if a != b]
    assert diff == [('MODE = "probe"', 'MODE = "exp"'), ("CONFIG = None", 'CONFIG = "e1c_seed3.yaml"')]
    # the other modes leave CONFIG = None
    for mode in ("e1", "e1b", "lrcheck", "measure", "probe"):
        q, o = _push(tmp_path / mode, "--mode", mode)
        assert q.returncode == 0, q.stderr
        assert "CONFIG = None" in (o / "kaggle_notebook.ipynb").read_text(encoding="utf-8")
    # refusals
    no_cfg = _push(tmp_path / "nocfg", "--mode", "exp")[0]
    assert no_cfg.returncode != 0 and "--config" in (no_cfg.stdout + no_cfg.stderr)
    for mode in ("e1", "probe", "infer", "lrcheck"):
        wrong = _push(tmp_path / ("wrong_" + mode), "--mode", mode, "--config", "e1c_seed3.yaml", "--run-id", "x")[0]
        assert wrong.returncode != 0, mode
    for bad in ("nope.yaml", "e1c_seed3", "../configs/e1c_seed3.yaml", "base.txt", "sub/e1c_seed3.yaml"):
        assert _push(tmp_path / ("bad_" + bad.replace("/", "_").replace(".", "_")), "--mode", "exp", "--config", bad)[0].returncode != 0, bad
    # the budget rules apply to exp as to the other training modes
    assert _push(tmp_path / "steps", "--mode", "exp", "--config", "e1c_seed3.yaml", "--steps", "4000")[0].returncode != 0
    same = _push(tmp_path / "same", "--mode", "exp", "--config", "e1c_seed3.yaml", "--steps", "7000")[0]
    assert same.returncode == 0, same.stderr
    d, o = _push(tmp_path / "dry", "--mode", "exp", "--config", "e6a_flip_rot90.yaml", "--steps", "6", "--dry", "--folds", "0")
    assert d.returncode == 0 and "OVERRIDE" in d.stdout
    assert "STEPS = 6" in (o / "kaggle_notebook.ipynb").read_text(encoding="utf-8")
    ia, o = _push(tmp_path / "ia", "--mode", "exp", "--config", "e4a_focal_dice.yaml", "--infer-after")
    assert ia.returncode == 0 and "INFER_AFTER = True" in (o / "kaggle_notebook.ipynb").read_text(encoding="utf-8")
    assert _push(tmp_path / "ia_lr", "--mode", "lrcheck", "--infer-after")[0].returncode != 0  # still refused elsewhere
    # an unset budget refuses exp as well
    assert prepare_push_folder.check_budget("exp", None, False, None)
    assert prepare_push_folder.check_budget("exp", 6, True, 7000) == []
    assert "exp" in prepare_push_folder.MODES and "exp" in prepare_push_folder.TRAINING_MODES


def test_exp_branch_of_the_notebook_runs_the_five_folds_of_its_config_and_e1_e1b_are_unchanged(tmp_path):
    """The notebook's own parameter, helper and dispatch cells are executed with stubs for the processes and the finish step:
    what is tested is which commands, run ids and arguments each mode produces."""
    import types

    from filament import runinfo
    from filament.config import load_config

    cells = _notebook_cells(K / "kaggle_notebook.ipynb")
    params = next(c for c in cells if "INPUT = Path(" in c)
    helper = next(c for c in cells if "def effective_steps" in c)
    dispatch = next(c for c in cells if "N_GPU = max(1, torch.cuda.device_count())" in c)
    budget_line = next(l for c in cells for l in c.splitlines() if l.startswith("BUDGET = load_config("))
    for old in ('WORK = Path("/kaggle/working")', 'TEMP = Path("/kaggle/temp")', 'INPUT = Path("/kaggle/input")'):
        assert params.count(old) == 1
    params = (params.replace('WORK = Path("/kaggle/working")', "WORK = Path(%r)" % str(tmp_path / "working"))
              .replace('TEMP = Path("/kaggle/temp")', "TEMP = Path(%r)" % str(tmp_path / "temp"))
              .replace('INPUT = Path("/kaggle/input")', "INPUT = Path(%r)" % str(tmp_path / "input")))

    def run_mode(mode, config=None, folds=None, infer_after=False, dry=False, steps=None):
        p = params.replace('MODE = "probe"', 'MODE = "%s"' % mode).replace("CONFIG = None", "CONFIG = %r" % config)
        p = p.replace("FOLDS = None", "FOLDS = %r" % (folds,)).replace("INFER_AFTER = False", "INFER_AFTER = %r" % infer_after)
        p = p.replace("DRY = False", "DRY = %r" % dry).replace("STEPS = None", "STEPS = %r" % (steps,))
        ns = {}
        exec(compile(p, "parameter cell", "exec"), ns)
        ns.update(REPO=REPO, load_config=load_config, runinfo=runinfo, CACHE=tmp_path / "cache", CACHE_WHY="test",
                  FOLDS_CSV=tmp_path / "folds.csv", torch=types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 2)))
        exec(compile(budget_line, "budget line", "exec"), ns)
        seen = {}

        def fake_run(jobs, n_gpu):
            seen["jobs"], seen["n_gpu"] = jobs, n_gpu
            return [(label, 0) for label, _ in jobs]

        def fake_finish(results, what, run_ids, infer=None):
            seen["finish"] = (what, run_ids, infer)

        ns["procs"] = types.SimpleNamespace(run_processes=fake_run)
        exec(compile(helper, "helper cell", "exec"), ns)
        ns["finish_training_session"] = fake_finish
        exec(compile(dispatch, "dispatch cell", "exec"), ns)
        return ns, seen

    def command_facts(cmd):
        return {"config": Path(cmd[cmd.index("--config") + 1]).name, "fold": int(cmd[cmd.index("--fold") + 1]),
                "steps": int(cmd[cmd.index("--steps") + 1]), "lr": "--lr" in cmd, "seed": "--seed" in cmd, "dry": "--mark-dry" in cmd}

    # exp: five folds of the named config, its own budget and run id, no lr / seed override, inference only when asked
    ns, seen = run_mode("exp", config="e6b_flip_rot90_brightness.yaml")
    assert ns["BUDGET"] == 7000 and seen["n_gpu"] == 2
    assert [l for l, _ in seen["jobs"]] == ["exp e6b_flip_rot90_brightness.yaml fold %d" % k for k in range(5)]
    assert [command_facts(c) for _, c in seen["jobs"]] == [
        {"config": "e6b_flip_rot90_brightness.yaml", "fold": k, "steps": 7000, "lr": False, "seed": False, "dry": False} for k in range(5)]
    assert seen["finish"] == ("exp e6b_flip_rot90_brightness.yaml", ["E6b_r512_lr1e-3_seed1"], None)
    ns, seen = run_mode("exp", config="e1c_seed3.yaml", folds=[0, 2], infer_after=True)
    assert [command_facts(c)["fold"] for _, c in seen["jobs"]] == [0, 2]
    assert seen["finish"] == ("exp e1c_seed3.yaml", ["E1c_r512_lr1e-3_seed3"], ("e1c_seed3.yaml", "E1c_r512_lr1e-3_seed3"))
    ns, seen = run_mode("exp", config="e4b_tversky.yaml", dry=True, steps=6, folds=[0])
    assert command_facts(seen["jobs"][0][1]) == {"config": "e4b_tversky.yaml", "fold": 0, "steps": 6, "lr": False, "seed": False, "dry": True}
    assert seen["finish"][1] == ["DRY_E4b_r512_lr1e-3_seed1"]
    with pytest.raises(AssertionError, match="approved budget 7000"):  # another number without DRY is refused
        run_mode("exp", config="e1c_seed3.yaml", steps=4000)
    # e1 and e1b behave as before (same commands, run ids and finish arguments as the lines of their branch always produced)
    for mode, cfg, rid in (("e1", "e1_seed1.yaml", "E1_r512_lr1e-3_seed1"), ("e1b", "e1b_seed2.yaml", "E1b_r512_lr1e-3_seed2")):
        ns, seen = run_mode(mode, infer_after=True)
        assert [l for l, _ in seen["jobs"]] == ["%s fold %d" % (mode, k) for k in range(5)]
        assert [command_facts(c) for _, c in seen["jobs"]] == [
            {"config": cfg, "fold": k, "steps": 7000, "lr": False, "seed": False, "dry": False} for k in range(5)]
        assert seen["finish"] == (mode, [rid], (cfg, rid))
        assert ns["BUDGET"] == 7000
    # CONFIG is set exactly when MODE = "exp", and it is a bare file name
    for mode, cfg in (("exp", None), ("e1", "e1c_seed3.yaml"), ("probe", "e1c_seed3.yaml"), ("exp", "../x.yaml"),
                      ("exp", "sub/x.yaml"), ("exp", "x.txt")):
        with pytest.raises(AssertionError):
            run_mode(mode, config=cfg)
