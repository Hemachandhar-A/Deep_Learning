"""Five-fold split, every image with all its versions in one fold, stratified by instrument (Arch section 4.1, D9).

Why grouped: an image has 1 to 3 annotator versions (S3); all versions of one image must be on the same side of a
split, otherwise the held-out score leaks.  Why stratified by instrument: the six GONG sites are present in similar
numbers in train and test (A4.1), so every fold should contain all six in similar proportions.

Decided by the user on 2026-10-12 after the Gate 1 review (D-a): **strategy ``"balanced"``, seed 42**, fixed before any
model result exists.  The result is written once as ``folds.csv`` (image, instrument, n_versions, fold) and reused by
every experiment so that arms can be compared pairwise (A4.1, A4.8 item 5).  ``folds.csv`` lists competition file names,
so it is *not* committed (D-b); its SHA-256 and the scikit-learn version are recorded with every run
(``provenance``), and the file ships in the private Kaggle dataset.

Two strategies (``strategy=``):

* ``"balanced"`` (default): rows are images, label = instrument x number of versions, ``StratifiedKFold`` (shuffle on).
  One row per image means every image, with all its versions, is in exactly one fold.  Spread over 100 seeds: per-site
  image counts differ by 2 between folds and fold version totals by 6, for every seed.  The reviewer measured the
  *identical* seed-42 split and the same spreads under scikit-learn 1.6.1 and 1.9.1 (Gate 1 review, F2).
* ``"document"``: rows are versions, group = image, label = instrument, ``StratifiedGroupKFold``; the literal first
  reading of A4.1.  Kept only to document the comparison.  It depends on the scikit-learn version: under 1.6.1 the
  images of one site per fold differ by 9 to 22 over 100 seeds; scikit-learn 1.8.0 fixed ``shuffle=True`` stratification
  of ``StratifiedGroupKFold`` (PR 32540), so under 1.9.1 the spread is 1-2 but only 136 of 707 images keep their fold
  at seed 42.  Do not use it for runs.
"""
from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .data import instrument_of
from .io_utils import atomic_open

FOLDS_COLUMNS = ("image", "instrument", "n_versions", "fold")
DEFAULT_STRATEGY = "balanced"  # decided by the user, 2026-10-12 (Gate 1 review, D-a)
DEFAULT_SEED = 42  # decided by the user, 2026-10-12 (Gate 1 review, D-a)


def make_folds(
    versions_of: Mapping[str, Sequence[str]],
    n_splits: int = 5,
    seed: int = DEFAULT_SEED,
    strategy: str = DEFAULT_STRATEGY,
) -> list[dict]:
    """Assign every image to one of ``n_splits`` folds.

    ``versions_of`` maps image name -> its version ids.  Returns one dict per image, sorted by image name.
    """
    if strategy not in ("document", "balanced"):
        raise ValueError("strategy must be 'document' or 'balanced', got %r" % (strategy,))
    image_codes = {name: i for i, name in enumerate(sorted(versions_of))}
    instruments = sorted({instrument_of(n) for n in versions_of})
    inst_code = {s: i for i, s in enumerate(instruments)}
    rows_image: list[str] = []
    for name in sorted(versions_of):
        # document: one row per version; balanced: one row per image
        rows_image += [name] * (len(versions_of[name]) if strategy == "document" else 1)
    if strategy == "document":
        y = np.array([inst_code[instrument_of(n)] for n in rows_image])
    else:
        y = np.array([inst_code[instrument_of(n)] * 10 + len(versions_of[n]) for n in rows_image])
    groups = np.array([image_codes[n] for n in rows_image])
    X = np.zeros((len(rows_image), 1))
    fold_of_image: dict[str, int] = {}
    if strategy == "document":
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    split_args = (X, y, groups) if strategy == "document" else (X, y)  # one row per image: groups are implicit
    for fold, (_, held_out) in enumerate(splitter.split(*split_args)):
        for i in held_out:
            name = rows_image[i]
            if fold_of_image.setdefault(name, fold) != fold:
                raise RuntimeError("versions of %s landed in different folds" % name)
    return [
        {"image": n, "instrument": instrument_of(n), "n_versions": len(versions_of[n]), "fold": fold_of_image[n]}
        for n in sorted(versions_of)
    ]


def round_robin_folds(versions_of: Mapping[str, Sequence[str]], n_splits: int = 2) -> list[dict]:
    """Deterministic folds for *dry runs on a handful of images only* (fewer images than ``StratifiedKFold`` needs).

    Images are sorted by (instrument, name) and dealt to the folds in turn, so every instrument is spread over the folds.
    It is not the experimental split and is never written as ``folds.csv``.
    """
    order = sorted(versions_of, key=lambda n: (instrument_of(n), n))
    return [
        {"image": n, "instrument": instrument_of(n), "n_versions": len(versions_of[n]), "fold": i % n_splits}
        for i, n in enumerate(order)
    ]


def write_folds_csv(path: str | Path, rows: Sequence[dict]) -> None:
    """Write folds.csv with ``\\n`` line endings on every platform (so the SHA-256 does not depend on the OS)."""
    with atomic_open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(FOLDS_COLUMNS)
        for r in rows:
            w.writerow([r["image"], r["instrument"], r["n_versions"], r["fold"]])


def read_folds_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != FOLDS_COLUMNS:
            raise ValueError("unexpected folds.csv header %r" % (reader.fieldnames,))
        return [
            {"image": r["image"], "instrument": r["instrument"], "n_versions": int(r["n_versions"]), "fold": int(r["fold"])}
            for r in reader
        ]


def file_sha256(path: str | Path) -> str:
    """SHA-256 of the file bytes (hex)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def provenance(path: str | Path) -> dict:
    """What every run records about its folds (D-b): the file's SHA-256 and the scikit-learn version in use.

    The scikit-learn version is that of the *current* environment; for the ``balanced`` strategy the split does not
    depend on it (Gate 1 review, F2), and the hash proves the file is the same.
    """
    return {"folds_csv_sha256": file_sha256(path), "scikit_learn_version": sklearn.__version__}


def verify_folds_file(path: str | Path, expected_sha256: str | None) -> str:
    """Return the file's SHA-256; raise if ``expected_sha256`` is given and differs (a run must not use another split)."""
    got = file_sha256(path)
    if expected_sha256 and got != expected_sha256:
        raise ValueError("folds file %s has SHA-256 %s, expected %s" % (path, got, expected_sha256))
    return got


class FoldsVerificationError(RuntimeError):
    """The folds file is not the one the experiment was defined with (or there is nothing to check it against)."""


def sidecar_path(folds_path: str | Path) -> Path:
    return Path(str(folds_path) + ".provenance.json")


def read_sidecar(folds_path: str | Path) -> dict | None:
    p = sidecar_path(folds_path)
    if not p.exists():
        return None
    import json

    return json.loads(p.read_text(encoding="utf-8"))


def verify_for_run(folds_path: str | Path, configured_sha256: str | None = None, allow_unverified: bool = False) -> dict:
    """Gate 1b review B1: the check every run makes *before* it uses ``folds.csv``.

    The expected hash comes from the configuration (``folds.sha256``, normally the git-ignored ``configs/local*.yaml``)
    and/or the sidecar written next to the file by ``scripts/make_folds.py``.  Rules:

    * a hash is expected and differs from the file's  -> :class:`FoldsVerificationError`, always;
    * two expected hashes (config and sidecar) disagree -> :class:`FoldsVerificationError`;
    * no expected hash exists                          -> :class:`FoldsVerificationError`, unless ``allow_unverified``
      (a flag for tests and dry runs only; the result then says ``verified: False``).

    Returns what ``metrics.json`` records: the file's SHA-256, the scikit-learn version in use, and strategy and seed
    from the sidecar when it exists.
    """
    folds_path = Path(folds_path)
    if not folds_path.exists():
        raise FileNotFoundError("folds file not found: %s" % folds_path)
    actual = file_sha256(folds_path)
    side = read_sidecar(folds_path)
    side_hash = side.get("folds_csv_sha256") if side else None
    if configured_sha256 and side_hash and configured_sha256 != side_hash:
        raise FoldsVerificationError(
            "the expected hash of the config (%s...) and of the sidecar (%s...) disagree" % (configured_sha256[:12], side_hash[:12])
        )
    expected = configured_sha256 or side_hash
    source = "config" if configured_sha256 else ("sidecar" if side_hash else None)
    if expected and actual != expected:
        raise FoldsVerificationError(
            "folds file has SHA-256 %s... but %s expects %s...: it is not the split this experiment was defined with"
            % (actual[:12], source, expected[:12])
        )
    if not expected and not allow_unverified:
        raise FoldsVerificationError(
            "no expected SHA-256 for the folds file (set folds.sha256 in a git-ignored configs/local*.yaml, or keep "
            "folds.csv.provenance.json next to the file); --allow-unverified-folds is for tests and dry runs only"
        )
    return {
        "folds_csv_sha256": actual,
        "expected_sha256_source": source,
        "folds_verified": bool(expected),
        "scikit_learn_version": sklearn.__version__,
        "folds_strategy": side.get("strategy") if side else None,
        "folds_seed": side.get("seed") if side else None,
    }


def fold_indices(rows: Sequence[dict], image_names: Sequence[str], fold: int) -> tuple[list[int], list[int]]:
    """(train_idx, val_idx): positions in ``image_names`` (the cache order) of the images outside / inside ``fold``."""
    fold_of = {r["image"]: r["fold"] for r in rows}
    missing = [n for n in image_names if n not in fold_of]
    if missing:
        raise KeyError("images without a fold: %s" % missing[:5])
    val = [i for i, n in enumerate(image_names) if fold_of[n] == fold]
    train = [i for i, n in enumerate(image_names) if fold_of[n] != fold]
    return train, val


def check_folds(rows: Sequence[dict], versions_of: Mapping[str, Sequence[str]], n_splits: int = 5) -> list[str]:
    """Problems found (empty list = fine): every image once, a valid fold, versions counted consistently, all folds used."""
    problems: list[str] = []
    names = [r["image"] for r in rows]
    if len(names) != len(set(names)):
        problems.append("an image appears more than once")
    if set(names) != set(versions_of):
        problems.append("image set differs from the annotations")
    for r in rows:
        if not 0 <= r["fold"] < n_splits:
            problems.append("%s has fold %s" % (r["image"], r["fold"]))
        if r["image"] in versions_of and r["n_versions"] != len(versions_of[r["image"]]):
            problems.append("%s: n_versions mismatch" % r["image"])
    if {r["fold"] for r in rows} != set(range(n_splits)):
        problems.append("not every fold is used")
    return problems


def fold_table(rows: Sequence[dict]) -> dict:
    """Counts per fold: images, versions, and images per instrument."""
    t: dict[int, dict] = defaultdict(lambda: {"images": 0, "versions": 0, "by_instrument": Counter()})
    for r in rows:
        e = t[r["fold"]]
        e["images"] += 1
        e["versions"] += r["n_versions"]
        e["by_instrument"][r["instrument"]] += 1
    return {k: t[k] for k in sorted(t)}


def spreads(rows: Sequence[dict]) -> dict:
    """Balance numbers used in the reports: worst-site image spread across folds, and the version-total spread."""
    insts = sorted({r["instrument"] for r in rows})
    n_folds = max(r["fold"] for r in rows) + 1
    img = np.zeros((n_folds, len(insts)), dtype=int)
    ver = np.zeros(n_folds, dtype=int)
    for r in rows:
        img[r["fold"], insts.index(r["instrument"])] += 1
        ver[r["fold"]] += r["n_versions"]
    site = img.max(axis=0) - img.min(axis=0)
    return {"worst_site_image_spread": int(site.max()), "version_total_spread": int(ver.max() - ver.min())}
