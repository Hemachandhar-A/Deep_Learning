"""Run the training processes of a Kaggle session and make a crashed one impossible to miss (Phase 2.1, item R2).

The notebook starts one ``scripts/train.py`` process per fold (or per learning-rate run), at most one per visible GPU,
each with its own ``CUDA_VISIBLE_DEVICES``.  The earlier version returned the bare exit codes in the order in which the
processes happened to finish, so a code could not be matched to its fold, and the notebook printed them and went on: a
session in which every fold had crashed still ended as COMPLETE.  Now

* :func:`run_processes` returns ``[(label, exit_code), ...]`` in the order of the jobs, and prints
  ``GPU n: <command>`` when it starts a process, so the first two-card session shows which process got which card;
* :func:`raise_on_failure` raises :class:`ProcessFailures` naming every failed label and its code; the notebook calls it
  *after* it has written the metrics, the scores, the manifest and the listing of the output folder;
* :func:`raise_if_any` does the same for the failed processes **and** for problems found afterwards (scoring, manifest,
  inference after training), so that a session can finish all its work and still end in an error with every reason named;
* every training process gets ``PYTHONUNBUFFERED=1``: in the drill of 2026-10-16 the text of the child processes reached
  the Kaggle log only when they exited (block-buffered pipes), not while they trained (Phase 2.2).

No GPU is needed to test this module: any command line can be a job.
"""
from __future__ import annotations

import os
import subprocess
import time
from typing import Callable, Sequence


class ProcessFailures(RuntimeError):
    """One or more processes of a session exited with a non-zero code (or could not be started)."""


def run_processes(jobs: Sequence[tuple[str, Sequence[str]]], n_gpu: int = 1, log: Callable = print,
                  poll_seconds: float = 2.0, popen=subprocess.Popen) -> list[tuple[str, int]]:
    """Run ``jobs`` = ``[(label, command), ...]``; at most ``max(1, n_gpu)`` at a time, one per GPU index.

    Returns ``[(label, exit_code), ...]`` in the order of ``jobs``.  A command that cannot be started gets code 127
    (reported like any other failure); nothing is raised here."""
    n = max(1, int(n_gpu))
    results: dict[int, int] = {}
    running: list[tuple[int, int, object]] = []  # (gpu, job index, process)
    for i, (label, cmd) in enumerate(jobs):
        while len(running) >= n:
            for item in list(running):
                if item[2].poll() is not None:
                    results[item[1]] = int(item[2].returncode)
                    running.remove(item)
            if len(running) >= n:
                time.sleep(poll_seconds)
        used = {g for g, _, _ in running}
        gpu = next(g for g in range(n) if g not in used)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
        if log:
            log("GPU %d: %s   [%s]" % (gpu, " ".join(str(c) for c in list(cmd)[1:]), label))
        try:
            running.append((gpu, i, popen(list(cmd), env=env)))
        except OSError as e:
            if log:
                log("could not start %s: %s" % (label, e))
            results[i] = 127
    for _, i, p in running:
        results[i] = int(p.wait())
    return [(label, results[i]) for i, (label, _) in enumerate(jobs)]


def failed(results: Sequence[tuple[str, int]]) -> list[tuple[str, int]]:
    return [(label, code) for label, code in results if code != 0]


def raise_on_failure(results: Sequence[tuple[str, int]], what: str = "session") -> None:
    """Raise :class:`ProcessFailures` unless every process exited with code 0."""
    bad = failed(results)
    if bad:
        raise ProcessFailures("%s: %d of %d processes failed: %s" % (
            what, len(bad), len(results), ", ".join("%s (exit code %d)" % (label, code) for label, code in bad)))


def raise_if_any(results: Sequence[tuple[str, int]], problems: Sequence[str] = (), what: str = "session") -> None:
    """Raise :class:`ProcessFailures` if a process failed or a later step recorded a problem; both are named."""
    bad = failed(results)
    if not bad and not problems:
        return
    parts = []
    if bad:
        parts.append("%d of %d processes failed: %s" % (
            len(bad), len(results), ", ".join("%s (exit code %d)" % (label, code) for label, code in bad)))
    if problems:
        parts.append("%d problem(s) after the processes returned: %s" % (len(problems), "; ".join(problems)))
    raise ProcessFailures("%s: %s" % (what, " | ".join(parts)))
