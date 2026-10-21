"""Predict the emotion of an audio file or a microphone recording."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aum2.cli.predict import main

if __name__ == "__main__":
    raise SystemExit(main())
