"""Atomic writes (review P3 / B3)."""
import hashlib
import json
import os

import numpy as np
import pytest

from filament import io_utils as io


def _leftovers(folder):
    return [p.name for p in folder.iterdir() if p.name.endswith(".tmp")]


def test_write_replaces_atomically_and_leaves_no_temporary(tmp_path):
    p = tmp_path / "metrics.json"
    io.atomic_write_json(p, {"b": 2, "a": 1})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert p.read_text(encoding="utf-8").endswith("\n")
    io.atomic_write_json(p, {"a": 3})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 3}
    assert _leftovers(tmp_path) == []


def test_failure_in_the_middle_keeps_the_old_file_and_removes_the_temporary(tmp_path):
    p = tmp_path / "ckpt.bin"
    io.atomic_write_bytes(p, b"old content")
    with pytest.raises(RuntimeError):
        with io.atomic_open(p, "wb") as f:
            f.write(b"half of the new con")
            raise RuntimeError("the machine crashed here")
    assert p.read_bytes() == b"old content"
    assert _leftovers(tmp_path) == []


def test_new_file_does_not_appear_when_the_block_fails(tmp_path):
    p = tmp_path / "new.npy"
    with pytest.raises(ValueError):
        with io.atomic_open(p, "wb") as f:
            f.write(b"x")
            raise ValueError("boom")
    assert not p.exists() and _leftovers(tmp_path) == []


def test_text_uses_lf_and_utf8_and_creates_folders(tmp_path):
    p = tmp_path / "a" / "b" / "t.txt"
    io.atomic_write_text(p, "line1\nline2é\n")
    assert p.read_bytes() == "line1\nline2é\n".encode("utf-8")  # no CRLF translation on Windows


def test_npy_and_torch_round_trip_and_checksum(tmp_path):
    a = np.arange(12, dtype=np.float16).reshape(3, 4)
    p = tmp_path / "maps.npy"
    io.atomic_save_npy(p, a)
    assert np.array_equal(np.load(p), a)
    assert io.sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()

    import torch

    q = tmp_path / "w.pt"
    io.atomic_torch_save(q, {"w": torch.arange(4.0)})
    assert torch.equal(torch.load(q, weights_only=True)["w"], torch.arange(4.0))


def test_finalise_file_renames_after_fsync(tmp_path):
    t, f = tmp_path / "x.npy.tmp", tmp_path / "x.npy"
    t.write_bytes(b"abc")
    io.finalise_file(t, f)
    assert f.read_bytes() == b"abc" and not t.exists()


def test_canonical_json_hash_is_order_independent():
    assert io.canonical_json_sha256({"a": 1, "b": [1, 2]}) == io.canonical_json_sha256({"b": [1, 2], "a": 1})
    assert io.canonical_json_sha256({"a": 1}) != io.canonical_json_sha256({"a": 2})


def test_fsync_is_called_before_the_rename(tmp_path, monkeypatch):
    order = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (order.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(os, "replace", lambda a, b: (order.append("replace"), real_replace(a, b))[1])
    io.atomic_write_bytes(tmp_path / "f.bin", b"data")
    assert order[0] == "fsync" and "replace" in order and order.index("fsync") < order.index("replace")


def test_directory_fsync_error_is_ignored_but_the_file_fsync_and_rename_stay_strict(tmp_path, monkeypatch):
    """Gate 2 finding G2-3 (Phase 2.1): some file systems refuse fsync on a directory; that alone must not fail a write."""
    real_open, real_fsync = os.open, os.fsync
    # 1. the directory part on its own: opening or fsyncing the directory raises OSError -> ignored
    monkeypatch.setattr(os, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("cannot open a directory")))
    io._fsync_dir_posix(tmp_path)
    monkeypatch.setattr(os, "open", real_open)
    closed = []
    monkeypatch.setattr(os, "open", lambda *a, **k: 123456)
    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("fsync on a directory refused")))
    monkeypatch.setattr(os, "close", lambda fd: closed.append(fd))
    io._fsync_dir_posix(tmp_path)  # no exception
    assert closed == [123456]  # and the descriptor was closed
    monkeypatch.undo()
    # 2. a failing *file* fsync is not ignored: the old file stays, no temporary file is left
    target = tmp_path / "keep.bin"
    target.write_bytes(b"old")

    def boom(fd):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError, match="disk full"):
        io.atomic_write_bytes(target, b"new")
    monkeypatch.setattr(os, "fsync", real_fsync)
    assert target.read_bytes() == b"old" and not list(tmp_path.glob("*.tmp"))
    # 3. a failing rename is not ignored either
    monkeypatch.setattr(os, "replace", lambda a, b: (_ for _ in ()).throw(OSError("rename refused")))
    with pytest.raises(OSError, match="rename refused"):
        io.atomic_write_bytes(target, b"new")
    monkeypatch.undo()
    assert target.read_bytes() == b"old" and not list(tmp_path.glob("*.tmp"))
