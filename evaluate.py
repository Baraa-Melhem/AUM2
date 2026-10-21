"""Recompute metrics and figures, build ensembles and the benchmark table."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aum2.cli.evaluate import main

if __name__ == "__main__":
    raise SystemExit(main())
