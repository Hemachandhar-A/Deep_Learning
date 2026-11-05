"""scripts/session_check.py (Gate 1b review, A1): detects a changed file, an untracked file, a NUL-filled file and a
broken ref, on a throw-away repository."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "session_check.py"
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(repo, *a):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.org", *a],
                   check=True, capture_output=True)


def _run(repo, *extra):
    p = subprocess.run([sys.executable, str(SCRIPT), "--repo", str(repo), *extra], capture_output=True, text=True)
    return p.returncode, p.stdout


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    _git(r, "init", "-q")
    (r / "a.py").write_text("print('a')\n", encoding="utf-8", newline="\n")
    (r / "b.md").write_text("# b\n", encoding="utf-8", newline="\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "init")
    return r


def test_clean_repository_passes(repo):
    rc, out = _run(repo)
    assert rc == 0 and "RESULT: OK" in out


def test_expect_and_manifest(repo, tmp_path):
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert _run(repo, "--expect", head[:8])[0] == 0
    assert _run(repo, "--expect", "deadbeef")[0] == 1
    import hashlib

    m = tmp_path / "SUMS.txt"
    m.write_text("%s  a.py\n%s  b.md\n" % (hashlib.sha256((repo / "a.py").read_bytes()).hexdigest(),
                                           hashlib.sha256((repo / "b.md").read_bytes()).hexdigest()), encoding="utf-8")
    assert _run(repo, "--manifest", str(m))[0] == 0
    m.write_text("%s  a.py\n" % ("0" * 64), encoding="utf-8")
    rc, out = _run(repo, "--manifest", str(m))
    assert rc == 1 and "a.py" in out


def test_modified_and_untracked_files_are_reported(repo):
    (repo / "a.py").write_text("print('changed')\n", encoding="utf-8", newline="\n")
    rc, out = _run(repo)
    assert rc == 1 and "a.py" in out
    _git(repo, "checkout", "--", "a.py")
    (repo / "new.txt").write_text("x\n", encoding="utf-8")
    rc, out = _run(repo)
    assert rc == 1 and "new.txt" in out


def test_nul_filled_file_and_zeroed_ref_are_reported(repo):
    (repo / "b.md").write_bytes(b"\x00" * 4)  # same kind of damage as the zero-filled files of 2026-10-12
    rc, out = _run(repo)
    assert rc == 1 and "NUL-damaged" in out
    _git(repo, "checkout", "--", "b.md")
    assert _run(repo)[0] == 0
    ref = repo / ".git" / "refs" / "heads"
    branch = next(p for p in ref.iterdir())
    branch.write_bytes(b"\x00" * 41)
    rc, out = _run(repo)
    assert rc == 1 and ("UNREADABLE" in out or "NUL-damaged" in out)


def _load_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("session_check_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_git_call_uses_no_optional_locks(repo, monkeypatch):
    """Review P1: `git status` must not refresh (write) the index, so every call carries --no-optional-locks."""
    mod = _load_module()
    calls = []
    real_run = subprocess.run

    def spy(cmd, *a, **k):
        if cmd and cmd[0] == "git":
            calls.append(list(cmd))
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(mod.subprocess, "run", spy)
    mod.git(repo, "status", "--short")
    mod.git(repo, "rev-parse", "HEAD")
    assert len(calls) == 2
    for c in calls:
        assert c[:2] == ["git", "--no-optional-locks"] and "-C" in c
    # and the whole routine, through main(): every recorded git call is guarded
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["session_check.py", "--repo", str(repo)])
    assert mod.main() == 0
    assert calls and all(c[1] == "--no-optional-locks" for c in calls)


def test_boot_time_is_reported_and_never_fails(repo):
    mod = _load_module()
    t = mod.boot_time()
    assert isinstance(t, str) and t
    rc, out = _run(repo)
    assert "boot time of this computer:" in out and rc == 0
