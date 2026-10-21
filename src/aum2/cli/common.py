"""Helpers shared by the command-line entry points."""

from __future__ import annotations

import argparse
import warnings


def add_override_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--set",
        dest="overrides",
        metavar="KEY=VALUE",
        nargs="+",
        default=[],
        help="override config values, e.g. --set train.epochs=20 split.n_folds=3",
    )


def quiet_warnings() -> None:
    """Silence third-party deprecation chatter so the structured log stays readable."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
