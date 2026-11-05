"""runinfo.py helpers added in Phase 2.1: cache location (never assumes /kaggle/temp), the label recorded in run
records, and the offline-weights folder inside Kaggle's unpacked layout (finding G2-13)."""
import shutil
from collections import namedtuple
from pathlib import Path

from filament import runinfo


def test_cache_goes_to_temp_only_when_it_exists_is_writable_and_has_room(tmp_path, monkeypatch):
    work, temp = tmp_path / "working", tmp_path / "temp"
    work.mkdir()
    # 1. no temp folder: the working folder, and the reason says why
    path, why = runinfo.choose_cache_dir(work, temp)
    assert path == work / "cache" and "does not exist" in why and not temp.exists()
    # 2. usable temp folder
    temp.mkdir()
    path, why = runinfo.choose_cache_dir(work, temp, min_free_gib=0.0)
    assert path == temp / "cache" and path.is_dir() and "writable" in why and not (path / ".write_test").exists()
    # 3. too little room, then enough
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(10 * 2**30, 9 * 2**30, 1 * 2**30))
    path, why = runinfo.choose_cache_dir(work, temp, min_free_gib=3.0)
    assert path == work / "cache" and "only 1.0 GiB free" in why
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(10 * 2**30, 1 * 2**30, 9 * 2**30))
    assert runinfo.choose_cache_dir(work, temp, min_free_gib=3.0)[0] == temp / "cache"
    # 4. a "folder" that is a file: fallback, no exception
    f = tmp_path / "afile"
    f.write_text("x")
    path, why = runinfo.choose_cache_dir(work, f)
    assert path == work / "cache" and "does not exist" in why


def test_unwritable_temp_folder_falls_back(tmp_path, monkeypatch):
    work, temp = tmp_path / "working", tmp_path / "temp"
    work.mkdir()
    temp.mkdir()

    def refuse(self, *a, **k):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(Path, "write_bytes", refuse)
    path, why = runinfo.choose_cache_dir(work, temp, min_free_gib=0.0)
    assert path == work / "cache" and "not usable" in why and "read-only" in why


def test_cache_location_label_names_only_the_two_kaggle_places(tmp_path):
    assert runinfo.cache_location_label("/kaggle/temp/cache") == "/kaggle/temp/cache"
    assert runinfo.cache_location_label(Path("/kaggle/working/cache/")) == "/kaggle/working/cache"
    assert runinfo.cache_location_label(tmp_path / "cache") == "other"  # a machine path is never recorded
    assert runinfo.cache_location_label(None) is None


def _model_folder(base: Path) -> Path:
    m = base / "models--smp-hub--resnet34.imagenet" / "snapshots" / "7a57b34f723329ff020b3f8bc41771163c519d0c"
    m.mkdir(parents=True)
    (m / "config.json").write_text("{}")
    return base


def test_hub_folder_is_found_in_kaggles_nested_layout_and_in_the_flat_one(tmp_path):
    # what Kaggle made of hf_cache.zip on 2026-10-14: <dataset>/hf_cache/hf_cache/hub/models--...
    nested = tmp_path / "ds1"
    _model_folder(nested / "hf_cache" / "hf_cache" / "hub")
    (nested / "code" / "repo").mkdir(parents=True)
    assert runinfo.find_hf_hub(nested) == nested / "hf_cache" / "hf_cache" / "hub"
    assert runinfo.find_hf_hub(nested / "hf_cache") == nested / "hf_cache" / "hf_cache" / "hub"
    # the zip extracted by the notebook itself: <work>/hf_cache/hub/models--...
    flat = tmp_path / "ws"
    _model_folder(flat / "hf_cache" / "hub")
    assert runinfo.find_hf_hub(flat / "hf_cache") == flat / "hf_cache" / "hub"
    assert runinfo.find_hf_hub(flat / "hf_cache" / "hub") == flat / "hf_cache" / "hub"
    # the shallowest hub with a model folder wins; an empty hub folder is not a hub
    both = tmp_path / "ds2"
    (both / "a" / "hub").mkdir(parents=True)
    _model_folder(both / "b" / "c" / "hub")
    assert runinfo.find_hf_hub(both) == both / "b" / "c" / "hub"
    # nothing there
    empty = tmp_path / "ds3"
    (empty / "hf_cache").mkdir(parents=True)
    assert runinfo.find_hf_hub(empty) is None


def test_environment_info_carries_cpu_and_gpu_counts():
    env = runinfo.environment_info()
    assert env["cpu_count"] >= 1 and env["gpu_count"] == len(env["gpu_names"])
    assert env["packages"].get("torch") and env["python"]
