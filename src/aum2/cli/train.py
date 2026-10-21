"""``train.py``: cross-validate a model family and score its fold ensemble on the test speakers."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from aum2.cli.common import add_override_argument, quiet_warnings
from aum2.config import load_config
from aum2.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one experiment: K speaker-disjoint dev folds + held-out test speakers.",
    )
    parser.add_argument("--config", required=True, help="experiment YAML, e.g. configs/crnn.yaml")
    parser.add_argument("--folds", type=int, nargs="+", help="train only these dev folds")
    parser.add_argument("--name", help="run name (default: the config's `name`)")
    parser.add_argument("--jobs", type=int, default=-1, help="CPU workers for feature caching")
    parser.add_argument("--log-level", default="INFO")
    add_override_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    quiet_warnings()
    overrides = list(args.overrides)
    if args.folds:
        overrides.append(f"folds=[{','.join(map(str, args.folds))}]")
    if args.name:
        overrides.append(f"name={args.name}")
    cfg = load_config(args.config, overrides)
    setup_logging(args.log_level, cfg.run_dir / "train.log")

    from aum2.training.experiment import run_experiment  # heavy imports after logging is up

    run_experiment(cfg, n_jobs=args.jobs)
    return 0
