"""Make ``src/`` importable when a script is run as ``python scripts/<name>.py`` (no install step needed)."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
