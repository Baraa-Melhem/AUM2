"""Choose hyper-parameters by dev out-of-fold UAR; the test speakers are never touched."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aum2.cli.tune import main

if __name__ == "__main__":
    raise SystemExit(main())
