"""Atomic file writes (Gate 1b review B3 and Follow-up review P3; risk K15: this PC reboots without warning).

Every result file (checkpoints, metrics, probability maps, folds provenance, submission CSVs, cache files) is written by
the same four steps, so that a crash leaves either the complete old file or the complete new file, never a half-written
or NUL-filled one:

1. write to a temporary file *in the same folder* (same volume, so the final rename is atomic);
2. ``flush`` and ``os.fsync`` the temporary file (the data are on the disk, not only in the cache);
3. ``os.replace`` it over the final name (atomic on Windows and POSIX);
4. on POSIX also ``fsync`` the folder so that the rename itself is durable (Windows cannot open a folder this way);
   this step alone is best effort: an ``OSError`` of the directory ``fsync`` is ignored, nothing else is.

If anything raises, the temporary file is removed and the old file is untouched.  ``sha256_file`` gives the checksum
that checkpoints record in their ``.json`` sidecar and that resume checks.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def _fsync_dir_posix(folder: Path) -> None:
    """Best effort: make the rename itself durable.  An ``OSError`` here (some file systems refuse ``fsync`` on a
    directory) is ignored (Gate 2 review G2-3); the *file* ``fsync`` and the rename are not wrapped and stay strict."""
    try:
        fd = os.open(str(folder), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _fsync_dir(folder: Path) -> None:
    if os.name == "nt":  # a directory cannot be opened for fsync on Windows; NTFS journals the rename
        return
    _fsync_dir_posix(folder)


@contextmanager
def atomic_open(path: str | Path, mode: str = "wb", encoding: str | None = None, newline: str | None = None) -> Iterator:
    """``with atomic_open(p, 'wb') as f: f.write(...)``; the file appears at ``p`` only when the block ended cleanly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, mode, encoding=encoding, newline=newline) as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    with atomic_open(path, "wb") as f:
        f.write(data)


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """Text with ``\\n`` line endings on every platform (so checksums do not depend on the OS)."""
    with atomic_open(path, "w", encoding=encoding, newline="\n") as f:
        f.write(text)


def atomic_write_json(path: str | Path, obj: Any, indent: int = 1) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, sort_keys=True) + "\n")


def atomic_save_npy(path: str | Path, array: np.ndarray) -> None:
    with atomic_open(path, "wb") as f:
        np.save(f, array)


def atomic_torch_save(path: str | Path, obj: Any) -> None:
    import torch

    with atomic_open(path, "wb") as f:
        torch.save(obj, f)


def finalise_file(tmp: str | Path, final: str | Path) -> None:
    """For files written by other means (e.g. a memory-mapped ``.npy``): fsync ``tmp`` and rename it to ``final``."""
    tmp, final = Path(tmp), Path(final)
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    os.replace(tmp, final)
    _fsync_dir(final.parent)


def sha256_file(path: str | Path, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha256(obj: Any) -> str:
    """SHA-256 of the canonical JSON (sorted keys, no spaces) of ``obj``: a short identity of a configuration."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
