"""Turn a ``# %%``-cell Python source into an ``.ipynb`` (so the notebook is reviewed as plain text, built, not hand-edited).

    python kaggle/build_notebook.py <source.py> <output.ipynb>

A line ``# %%`` starts a code cell, ``# %% [markdown]`` a markdown cell whose lines are written as ``# text``.  The
notebook is written without outputs and without execution counts.  The cell sources are exactly the text of the source
file, so executing the source file as a script runs the same code as the notebook.
"""
import json
import sys
from pathlib import Path


def parse_cells(text: str) -> list[dict]:
    cells: list[dict] = []
    kind, buf = None, []

    def flush():
        nonlocal kind, buf
        if kind is None:
            return
        while buf and not buf[-1].strip():
            buf.pop()
        if buf:
            if kind == "markdown":
                src = [(l[2:] if l.startswith("# ") else l.lstrip("#")) for l in buf]
            else:
                src = list(buf)
            cells.append({"cell_type": kind, "metadata": {}, "source": [s + "\n" for s in src[:-1]] + [src[-1]],
                          **({"outputs": [], "execution_count": None} if kind == "code" else {})})
        kind, buf = None, []

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
        else:
            if kind is None:
                kind = "code"
            buf.append(line)
    flush()
    return cells


def build(src: Path, dst: Path) -> int:
    cells = parse_cells(src.read_text(encoding="utf-8"))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = "cell%03d" % i
    dst.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8", newline="\n")
    return len(cells)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    n = build(Path(sys.argv[1]), Path(sys.argv[2]))
    print("wrote %s (%d cells)" % (sys.argv[2], n))
