"""``tune.py``: grid search on the dev folds only (the test speakers are never touched)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import yaml

from aum2.cli.common import add_override_argument, quiet_warnings
from aum2.config import load_config
from aum2.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Choose hyper-parameters by dev out-of-fold UAR, e.g. "
        "--grid train.epochs=10,20,40 model.ssl.dropout=0.3,0.5",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--grid", nargs="+", default=[], metavar="KEY=V1,V2,...")
    parser.add_argument("--output", help="append the ranked table to this markdown file")
    parser.add_argument("--jobs", type=int, default=-1)
    add_override_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    quiet_warnings()
    setup_logging()
    from aum2.training.selection import grid_search

    base = load_config(args.config, args.overrides)
    grid = {}
    for item in args.grid:
        key, values = item.split("=", 1)
        grid[key] = [yaml.safe_load(v) for v in values.split(",")]
    table = grid_search(base, grid, args.jobs)
    metrics = ["war", "uar", "macro_f1"]
    table[metrics] = table[metrics].map(lambda value: f"{value:.4f}")
    markdown = table.to_markdown(index=False, disable_numparse=True)
    print(markdown)
    if args.output:
        header = f"\n### {Path(args.config).name} {' '.join(args.overrides)}\n\n"
        with Path(args.output).open("a", encoding="utf-8") as handle:
            handle.write(header + markdown + "\n")
    return 0
