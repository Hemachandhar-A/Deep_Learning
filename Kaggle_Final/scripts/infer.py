"""Command-line entry point of the test-set inference (see ``src/filament/infer.py``).

    python scripts/infer.py --config configs/e1_seed1.yaml --run-dir <work>/runs/<run_id> --cache-dir <cache> --out-dir <out>

Writes ``submission.csv`` (all checks run), the averaged float16 maps and ``infer_report.json``.  It never uploads.
"""
import _bootstrap  # noqa: F401

import sys

from filament.infer import main

if __name__ == "__main__":
    sys.exit(main())
