"""Session routine (Gate 1b review, section 5 A1): run at the start and at the end of every work session.

    python scripts/session_check.py [--expect <commit>] [--manifest <SHA256SUMS file>] [--also <folder> ...]

Prints and checks, read-only:

1. ``git status --short`` and ``git diff HEAD --stat`` (empty = nothing changed since the last commit);
2. ``git fsck --full`` (empty = the object database and refs are readable);
3. the commit at HEAD, and whether it is the commit you expect (``--expect``, a prefix is enough);
4. files whose bytes are all NUL, or that contain a run of 16 NUL bytes, in the repository (text files, and the small
   files of ``.git`` such as refs and COMMIT_EDITMSG) and in the ``--also`` folders;
5. with ``--manifest``: every tracked file against ``<sha256>  <path>`` lines (the ``SHA256SUMS`` of a gate package).

Also prints the Windows boot time (informational, never a failure): if it differs between the start and the end of a
session, the PC rebooted in between (risk K15) and the routine must be run again after the reboot.

Exit code 0 only when nothing unexpected was found.  A change you did not make: stop, show ``git diff``, tell the user,
do not build on it (review rule).  Every git call uses ``--no-optional-locks`` so that ``git status`` does not refresh
(write) the index; the script writes nothing to the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".csv", ".toml", ".cfg", ".ini", ".ipynb", ".sh", ".ps1"}
TEXT_NAMES = {".gitignore", ".gitattributes", "NOTICE", "LICENSE"}
SKIP_DIRS = {"__pycache__", ".pytest_cache", "objects", "venv", ".venv", "node_modules"}
RUN = b"\x00" * 16


GIT = ["git", "--no-optional-locks"]  # documented for scripts: `git status` must not take the index lock to refresh it


def git(repo: Path, *args: str) -> tuple[int, str]:
    p = subprocess.run([*GIT, "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout + p.stderr).strip()


def boot_time() -> str:
    """Time of the last boot of this computer, for the report; 'unknown' when it cannot be read."""
    try:
        if sys.platform == "win32":
            cmd = ["powershell", "-NoProfile", "-Command",
                   "(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')"]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
            return out or "unknown"
        import time

        up = float(Path("/proc/uptime").read_text().split()[0])
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - up))
    except Exception:
        return "unknown"


def nul_damaged(folder: Path, include_git_small_files: bool) -> list[str]:
    bad: list[str] = []
    for p in folder.rglob("*"):
        if not p.is_file() or any(part in SKIP_DIRS for part in p.parts):
            continue
        in_git = ".git" in p.parts
        if in_git and not include_git_small_files:
            continue
        if in_git and p.name == "index":
            continue  # the index legitimately contains NUL bytes
        if not in_git and p.suffix.lower() not in TEXT_SUFFIXES and p.name not in TEXT_NAMES:
            continue
        if p.stat().st_size == 0:
            continue
        b = p.read_bytes()
        if b.count(0) == len(b) or RUN in b:
            bad.append(str(p))
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--expect", default=None, help="commit (or prefix) that HEAD must equal")
    ap.add_argument("--manifest", type=Path, default=None, help="file with '<sha256>  <path>' lines for tracked files")
    ap.add_argument("--also", type=Path, nargs="*", default=[], help="extra folders to scan for NUL-damaged files")
    args = ap.parse_args()
    repo = args.repo
    problems: list[str] = []

    print("boot time of this computer: %s (informational; compare the start and the end of a session)" % boot_time())
    rc, head = git(repo, "rev-parse", "HEAD")
    print("HEAD: %s" % (head if rc == 0 else "UNREADABLE (%s)" % head))
    if rc != 0:
        problems.append("HEAD cannot be resolved")
    elif args.expect and not head.startswith(args.expect):
        problems.append("HEAD %s is not the expected commit %s" % (head[:10], args.expect))

    _, status = git(repo, "status", "--short")
    print("--- git status --short (empty = clean)\n%s" % status)
    if status:
        problems.append("working tree or index differs from HEAD / has untracked files")
    _, diffstat = git(repo, "diff", "HEAD", "--stat")
    print("--- git diff HEAD --stat (empty = no change)\n%s" % diffstat)
    if diffstat:
        problems.append("tracked files differ from HEAD")

    rc, fsck = git(repo, "fsck", "--full")
    print("--- git fsck --full (empty = clean)\n%s" % fsck)
    if rc != 0 or fsck:
        problems.append("git fsck reports problems")

    bad = nul_damaged(repo, include_git_small_files=True)
    for extra in args.also:
        bad += nul_damaged(extra, include_git_small_files=False)
    print("--- files that are all NUL or contain runs of NUL: %s" % (bad or "none"))
    if bad:
        problems.append("NUL-damaged files: %s" % bad)

    if args.manifest:
        sums = {}
        for line in args.manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                h, f = line.split("  ", 1)
                sums[f] = h
        differ = [f for f, h in sums.items()
                  if not (repo / f).exists() or hashlib.sha256((repo / f).read_bytes()).hexdigest() != h]
        print("--- manifest %s: %d files, differing or missing: %s" % (args.manifest.name, len(sums), differ or "none"))
        if differ:
            problems.append("files differ from the manifest: %s" % differ)

    print("RESULT: %s" % ("OK, nothing unexpected" if not problems else "PROBLEMS: " + "; ".join(problems)))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
