"""``evaluate.py``: recompute run reports, build ensembles, write the benchmark table."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from aum2.cli.common import quiet_warnings
from aum2.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained runs.")
    parser.add_argument("--run", nargs="+", default=[], help="run folder(s) to (re-)evaluate")
    parser.add_argument(
        "--ensemble", nargs="+", metavar="RUN", help="average these runs' probabilities"
    )
    parser.add_argument("--name", default="ensemble", help="output run name for --ensemble")
    parser.add_argument(
        "--tune-weights",
        action="store_true",
        help="choose ensemble weights on dev out-of-fold UAR (never on test)",
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        metavar="RUN",
        help="write the table, chart and per-run reports (list the selected model first)",
    )
    parser.add_argument("--output", default="reports/benchmark.md", help="where --benchmark writes")
    parser.add_argument(
        "--export",
        nargs="+",
        metavar="RUN",
        help="copy these runs' metrics, configs and figures into the reports folder (no table)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("RUN_A", "RUN_B"),
        help="paired speaker-bootstrap test of RUN_A minus RUN_B on the test speakers",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    quiet_warnings()
    setup_logging()
    from aum2.evaluation.report import (
        benchmark_table,
        compare_runs,
        ensemble_runs,
        evaluate_run,
        export_reports,
    )

    for run in args.run:
        evaluate_run(run)
    if args.ensemble:
        ensemble_runs(args.ensemble, args.name, Path(args.ensemble[0]).parent, args.tune_weights)
    if args.benchmark:
        table = benchmark_table(args.benchmark)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table + "\n", encoding="utf-8")
        export_reports(args.benchmark, output.parent)
        print(table)
    if args.export:
        export_reports(args.export, Path(args.output).parent, chart=False)
    if args.compare:
        result = compare_runs(*args.compare)
        for metric in ("uar", "war"):
            r = result[metric]
            print(
                f"{result['a']} - {result['b']} {metric.upper()}: {r['difference']:+.4f} "
                f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}], p = {r['p_value']:.4f}"
            )
    return 0
