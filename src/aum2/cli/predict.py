"""``predict.py``: emotion of an audio file (any length) or of microphone input."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from aum2.cli.common import quiet_warnings
from aum2.logging_utils import setup_logging

if TYPE_CHECKING:
    from aum2.inference import EmotionRecognizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict speech emotion with a trained run.")
    parser.add_argument("audio", nargs="*", default=[], help="audio file(s): wav, flac, ogg, ...")
    parser.add_argument("--mic", type=float, metavar="SECONDS", help="record from the microphone")
    parser.add_argument(
        "--loop", action="store_true", help="with --mic: keep listening, one result per chunk"
    )
    parser.add_argument(
        "--run", default="runs/wavlm", help="trained run folder (default: runs/wavlm)"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--window", type=float, default=3.0, help="window length for long audio (s)"
    )
    parser.add_argument("--hop", type=float, default=1.5, help="window hop for long audio (s)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    return parser


def _listen(recognizer: EmotionRecognizer, seconds: float, as_json: bool) -> None:
    """Record ``seconds`` chunks until Ctrl+C; audio never leaves the machine."""
    from aum2.inference import record_microphone

    print(f"Listening in {seconds:.0f} s chunks (Ctrl+C to stop) ...")
    try:
        while True:
            samples = record_microphone(seconds, recognizer.sample_rate)
            try:
                prediction = recognizer.predict_waveform(samples, recognizer.sample_rate)
            except ValueError:
                print("(silence)")
                continue
            if as_json:
                print(json.dumps(prediction.to_dict()))
            else:
                print(f"{prediction.label:<8} {prediction.confidence:.0%}")
    except KeyboardInterrupt:
        print("stopped")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if bool(args.audio) == (args.mic is not None):
        parser.error("give either audio file(s) or --mic SECONDS")
    if args.loop and args.mic is None:
        parser.error("--loop needs --mic SECONDS")
    quiet_warnings()
    setup_logging("WARNING")
    from aum2.inference import EmotionRecognizer, record_microphone

    recognizer = EmotionRecognizer(args.run, device=args.device)
    options = {"window_seconds": args.window, "hop_seconds": args.hop}
    if args.loop:
        _listen(recognizer, args.mic, args.json)
        return 0
    if args.mic is not None:
        print(f"Recording {args.mic:.0f} s ... speak now")
        samples = record_microphone(args.mic, recognizer.sample_rate)
        results = {
            "microphone": recognizer.predict_waveform(samples, recognizer.sample_rate, **options)
        }
    else:
        results = {path: recognizer.predict_file(path, **options) for path in args.audio}

    if args.json:
        print(json.dumps({k: v.to_dict() for k, v in results.items()}, indent=2))
        return 0
    for source, prediction in results.items():
        print(f"{source}: {prediction}")
        for seg in prediction.segments:
            print(
                f"    {seg.start:6.1f}-{seg.end:6.1f} s  {seg.label:<8}"
                f" {seg.probabilities[seg.label]:.0%}"
            )
    return 0
