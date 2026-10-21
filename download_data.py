"""Download the four corpora and build the manifest (add --prepare to cache audio)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aum2.cli.download import main

if __name__ == "__main__":
    raise SystemExit(main())
