"""Shared fixtures.  Tests that need the competition data are marked ``data`` and skipped when it cannot be found.

Point them at the data with ``FILAMENT_DATA_ROOT`` or ``--data-root`` (never hard-coded; Arch section 4.8 item 3).
The data are only read, never written.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pytest_addoption(parser):
    parser.addoption("--data-root", default=None, help="folder that contains the competition files")
    parser.addoption("--cache-dir", default=None, help="folder with the 1024 px cache (prepare_cache.py output)")


@pytest.fixture(scope="session")
def data_root(request):
    root = request.config.getoption("--data-root") or os.environ.get("FILAMENT_DATA_ROOT")
    if not root or not Path(root).exists():
        pytest.skip("competition data not available (set FILAMENT_DATA_ROOT or --data-root)")
    return Path(root)


@pytest.fixture(scope="session")
def layout(data_root):
    from filament import data

    return data.find_data_layout(data_root)


@pytest.fixture(scope="session")
def annotations(layout):
    from filament import data

    return data.load_annotations(layout.train_json)


@pytest.fixture(scope="session")
def cache_dir(request):
    d = request.config.getoption("--cache-dir") or os.environ.get("FILAMENT_CACHE_DIR")
    if not d or not (Path(d) / "cache_index.json").exists():
        pytest.skip("cache not built (run scripts/prepare_cache.py; set FILAMENT_CACHE_DIR or --cache-dir)")
    return Path(d)


@pytest.fixture(scope="session")
def gt_rles(annotations):
    """{version_id: [RLE, ...]} for all 1,154 versions (pycocotools rasterisation at 2048 px)."""
    from filament import data

    return {v: data.version_rles(a) for v, a in annotations.version_anns.items()}
