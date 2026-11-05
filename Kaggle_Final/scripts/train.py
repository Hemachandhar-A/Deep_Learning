"""Command-line entry point of the fold training (see ``src/filament/train.py`` for what it does).

    python scripts/train.py --config configs/e1_seed1.yaml --local-config configs/local.yaml --fold 0

One fold per process: set ``CUDA_VISIBLE_DEVICES`` to train two folds at once on two cards (A4.8 item 6; untested).
``--measure-only 60`` measures seconds per step including the data loader and prints the derived step budget; it never
changes a config.  Use ``--allow-unverified-folds`` only in tests and dry runs.
"""
import _bootstrap  # noqa: F401

import sys

from filament.train import main

if __name__ == "__main__":
    sys.exit(main())
