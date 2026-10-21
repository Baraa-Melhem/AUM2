"""``download_data.py``: fetch the corpora and (optionally) build the conditioned cache."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from aum2.cli.common import quiet_warnings
from aum2.config import CORPORA, DataConfig
from aum2.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download CREMA-D, RAVDESS, TESS and SAVEE.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--corpora", nargs="+", default=list(CORPORA), choices=CORPORA)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="also decode, resample and condition every clip into the cache",
    )
    parser.add_argument("--jobs", type=int, default=-1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    quiet_warnings()
    setup_logging()
    from aum2.data.dataset import build_manifest, load_corpus
    from aum2.data.download import download_all

    download_all(args.raw_dir, tuple(args.corpora))
    cfg = DataConfig(raw_dir=args.raw_dir, cache_dir=args.cache_dir, corpora=tuple(args.corpora))
    manifest = build_manifest(cfg.raw_dir, cfg.corpora, cfg.classes)
    table = manifest.groupby(["corpus", "emotion"]).size().unstack(fill_value=0)
    table["speakers"] = manifest.groupby("corpus")["speaker"].nunique()
    print(table.to_string())
    if args.prepare:
        load_corpus(cfg, n_jobs=args.jobs)
    return 0
