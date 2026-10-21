"""Cross-validate a model family and score its fold ensemble on the held-out test speakers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aum2.cli.train import main

if __name__ == "__main__":
    raise SystemExit(main())
