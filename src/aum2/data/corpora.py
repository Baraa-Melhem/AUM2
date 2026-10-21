"""Filename conventions of the four acted English emotion corpora.

Every parser looks at the *file name only* and must match it completely. The 2023 notebook
tested ``"a" in full_path`` and labelled every SAVEE clip "angry" because ``/kaggle/`` contains
an ``a``; anchored regular expressions on the bare name make that class of bug impossible.

Speakers are namespaced by corpus (``cremad_1001``), so speaker-disjoint splits stay correct
when corpora are pooled.
"""

from __future__ import annotations

import csv
import dataclasses
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from aum2.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Utterance:
    """One labelled recording; ``path`` is relative to the raw-data root, in POSIX form."""

    utt_id: str
    path: str
    corpus: str
    speaker: str
    gender: str
    emotion: str
    text_id: str
    intensity: str


class FilenameError(ValueError):
    """Raised when a file inside a corpus folder does not follow that corpus's convention."""


_CREMAD = re.compile(r"(?P<actor>\d{4})_(?P<text>[A-Z]{3})_(?P<emo>[A-Z]{3})_(?P<level>[A-Z]{2})")
_RAVDESS = re.compile(
    r"(?P<modality>\d{2})-(?P<channel>\d{2})-(?P<emo>\d{2})-(?P<level>\d{2})"
    r"-(?P<text>\d{2})-(?P<rep>\d{2})-(?P<actor>\d{2})"
)
_TESS = re.compile(r"(?P<speaker>OAF|YAF)_(?P<word>[a-z]+)_(?P<emo>[a-z]+)", re.IGNORECASE)
_SAVEE = re.compile(r"(?P<speaker>DC|JE|JK|KL)_(?P<emo>sa|su|a|d|f|h|n)(?P<idx>\d{2})")

CREMAD_EMOTIONS = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}
TESS_EMOTIONS = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "ps": "surprise",
}
SAVEE_EMOTIONS = {
    "a": "angry",
    "d": "disgust",
    "f": "fear",
    "h": "happy",
    "n": "neutral",
    "sa": "sad",
    "su": "surprise",
}


def _match(pattern: re.Pattern[str], path: Path) -> re.Match[str]:
    if path.suffix.lower() != ".wav" or not (match := pattern.fullmatch(path.stem)):
        raise FilenameError(f"unexpected file name {path.name!r}")
    return match


def parse_cremad(path: Path, sex_by_actor: Mapping[str, str]) -> Utterance:
    """``1001_DFA_ANG_XX.wav``: actor, sentence, emotion, intensity (LO/MD/HI/XX)."""
    m = _match(_CREMAD, path)
    if m["emo"] not in CREMAD_EMOTIONS:
        raise FilenameError(f"unknown CREMA-D emotion code in {path.name!r}")
    return Utterance(
        utt_id=f"cremad_{path.stem}",
        path="",
        corpus="cremad",
        speaker=f"cremad_{m['actor']}",
        gender=sex_by_actor.get(m["actor"], "unknown"),
        emotion=CREMAD_EMOTIONS[m["emo"]],
        text_id=m["text"],
        intensity=m["level"],
    )


def parse_ravdess(path: Path) -> Utterance | None:
    """``03-01-05-01-02-01-12.wav``; returns ``None`` for video or song files."""
    m = _match(_RAVDESS, path)
    if m["modality"] != "03" or m["channel"] != "01":
        return None
    if m["emo"] not in RAVDESS_EMOTIONS:
        raise FilenameError(f"unknown RAVDESS emotion code in {path.name!r}")
    actor = int(m["actor"])
    return Utterance(
        utt_id=f"ravdess_{path.stem}",
        path="",
        corpus="ravdess",
        speaker=f"ravdess_{actor:02d}",
        gender="male" if actor % 2 else "female",
        emotion=RAVDESS_EMOTIONS[m["emo"]],
        text_id=m["text"],
        intensity="normal" if m["level"] == "01" else "strong",
    )


def parse_tess(path: Path) -> Utterance:
    """``OAF_back_angry.wav``: older/younger actress, target word, emotion (``ps`` = surprise)."""
    m = _match(_TESS, path)
    emotion = m["emo"].lower()
    if emotion not in TESS_EMOTIONS:
        raise FilenameError(f"unknown TESS emotion in {path.name!r}")
    speaker = m["speaker"].upper()
    return Utterance(
        utt_id=f"tess_{speaker}_{m['word'].lower()}_{emotion}",
        path="",
        corpus="tess",
        speaker=f"tess_{speaker}",
        gender="female",
        emotion=TESS_EMOTIONS[emotion],
        text_id=m["word"].lower(),
        intensity="XX",
    )


def parse_savee(path: Path) -> Utterance:
    """``DC_sa01.wav``: one of four male speakers, emotion code, sentence index."""
    m = _match(_SAVEE, path)
    return Utterance(
        utt_id=f"savee_{path.stem}",
        path="",
        corpus="savee",
        speaker=f"savee_{m['speaker']}",
        gender="male",
        emotion=SAVEE_EMOTIONS[m["emo"]],
        text_id=f"{m['emo']}{m['idx']}",
        intensity="XX",
    )


def read_cremad_demographics(csv_path: Path) -> dict[str, str]:
    """Map CREMA-D ``ActorID`` to ``male``/``female`` from ``VideoDemographics.csv``."""
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["ActorID"].strip(): row["Sex"].strip().lower() for row in csv.DictReader(handle)
        }


# Where each corpus lives under the raw-data root, as produced by ``download_data.py``.
CORPUS_LAYOUT: dict[str, str] = {
    "cremad": "cremad/AudioWAV",
    "ravdess": "ravdess",
    "tess": "tess",
    "savee": "savee/AudioData",
}
EXPECTED_FILES: dict[str, int] = {"cremad": 7442, "ravdess": 1440, "tess": 2800, "savee": 480}


def iter_corpus(raw_dir: Path, corpus: str) -> Iterator[Utterance]:
    """Yield every parseable utterance of ``corpus`` (all emotions, before class filtering)."""
    root = raw_dir / CORPUS_LAYOUT[corpus]
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found - run `python download_data.py` first")
    parser: Callable[[Path], Utterance | None]
    if corpus == "cremad":
        sex = read_cremad_demographics(raw_dir / "cremad" / "VideoDemographics.csv")
        parser = lambda p: parse_cremad(p, sex)  # noqa: E731
    else:
        parser = {"ravdess": parse_ravdess, "tess": parse_tess, "savee": parse_savee}[corpus]
    files = sorted(root.rglob("*.wav"), key=lambda p: p.as_posix().lower())
    if len(files) != EXPECTED_FILES[corpus]:
        logger.warning(
            "%s: found %d wav files, expected %d", corpus, len(files), EXPECTED_FILES[corpus]
        )
    for file in files:
        utterance = parser(file)
        if utterance is not None:
            yield dataclasses.replace(utterance, path=file.relative_to(raw_dir).as_posix())
