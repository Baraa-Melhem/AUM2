"""Download the four corpora from their original (or documented mirror) sources.

| corpus  | source                                                    | licence          |
|---------|-----------------------------------------------------------|------------------|
| CREMA-D | github.com/CheyneyComputerScience/CREMA-D (Git LFS)       | ODbL 1.0         |
| RAVDESS | zenodo.org/records/1188976 (speech, audio-only)           | CC BY-NC-SA 4.0  |
| TESS    | borealisdata.ca doi:10.5683/SP2/E8H2MF (Univ. of Toronto) | CC BY-NC 4.0     |
| SAVEE   | huggingface.co/datasets/AbstractTTS/SAVEE (mirror)        | research use     |

Every step is idempotent: files that already exist are skipped, so an interrupted download can
simply be re-run. Revisions are pinned so that the manifest is reproducible.
"""

from __future__ import annotations

import io
import json
import shutil
import urllib.request
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from aum2.data.corpora import CORPUS_LAYOUT, EXPECTED_FILES
from aum2.logging_utils import get_logger

logger = get_logger(__name__)

CREMAD_REPO = "CheyneyComputerScience/CREMA-D"
CREMAD_REVISION = "1658cd342dff90010aa843eaeebd53610a08b1dc"
RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip"
TESS_URL = (
    "https://borealisdata.ca/api/access/dataset/:persistentId/?persistentId=doi:10.5683/SP2/E8H2MF"
)
SAVEE_URL = (
    "https://huggingface.co/datasets/AbstractTTS/SAVEE/resolve/"
    "e9fdcb16523f5ade2fdf02c9b1aeecbdbcf7a547/data/train-00000-of-00001.parquet"
)
_USER_AGENT = {"User-Agent": "aum2-downloader/2.0 (+https://github.com/Baraa-Melhem/AUM2)"}


def _fetch(url: str, timeout: float = 120.0) -> bytes:
    request = urllib.request.Request(url, headers=_USER_AGENT)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _download_file(url: str, target: Path, desc: str) -> Path:
    """Stream ``url`` to ``target`` (via a ``.part`` file so a crash never leaves a stub)."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers=_USER_AGENT)
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=desc, leave=False) as bar:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                bar.update(len(chunk))
    partial.replace(target)
    return target


def _safe_extract(archive: Path, destination: Path, keep: Callable[[str], bool]) -> int:
    """Extract members accepted by ``keep``, refusing any path that escapes ``destination``."""
    destination = destination.resolve()
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.is_dir() or not keep(member.filename):
                continue
            target = (destination / member.filename).resolve()
            if destination not in target.parents:
                raise RuntimeError(f"refusing to extract {member.filename!r} outside {destination}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            count += 1
    return count


def _count_wavs(folder: Path) -> int:
    return sum(1 for _ in folder.rglob("*.wav")) if folder.is_dir() else 0


def download_ravdess(raw_dir: Path) -> None:
    folder = raw_dir / CORPUS_LAYOUT["ravdess"]
    if _count_wavs(folder) >= EXPECTED_FILES["ravdess"]:
        return
    archive = _download_file(RAVDESS_URL, folder / "Audio_Speech_Actors_01-24.zip", "RAVDESS")
    _safe_extract(archive, folder, keep=lambda name: name.lower().endswith(".wav"))
    archive.unlink()


def download_tess(raw_dir: Path) -> None:
    folder = raw_dir / CORPUS_LAYOUT["tess"]
    if _count_wavs(folder) >= EXPECTED_FILES["tess"]:
        return
    archive = _download_file(TESS_URL, folder / "tess.zip", "TESS")
    _safe_extract(archive, folder, keep=lambda name: name.lower().endswith(".wav"))
    archive.unlink()


def download_savee(raw_dir: Path) -> None:
    """The mirror stores the original ``DC_a01.wav`` files as bytes inside one parquet file."""
    import pyarrow.parquet as pq  # local import: only this corpus needs pyarrow

    folder = raw_dir / CORPUS_LAYOUT["savee"]
    if _count_wavs(folder) >= EXPECTED_FILES["savee"]:
        return
    table = pq.read_table(io.BytesIO(_fetch(SAVEE_URL, timeout=600)), columns=["file", "audio"])
    folder.mkdir(parents=True, exist_ok=True)
    for name, audio in zip(
        table.column("file").to_pylist(), table.column("audio").to_pylist(), strict=True
    ):
        if Path(name).name != name or not name.lower().endswith(".wav"):
            raise RuntimeError(f"unexpected SAVEE member name {name!r}")
        target = folder / name
        if not target.exists():
            target.write_bytes(audio["bytes"])


def download_cremad(raw_dir: Path, workers: int = 16) -> None:
    """List ``AudioWAV`` through the GitHub tree API and fetch each file from the LFS media host."""
    corpus_root = raw_dir / "cremad"
    folder = raw_dir / CORPUS_LAYOUT["cremad"]
    demographics = corpus_root / "VideoDemographics.csv"
    raw_base = f"https://raw.githubusercontent.com/{CREMAD_REPO}/{CREMAD_REVISION}"
    _download_file(f"{raw_base}/VideoDemographics.csv", demographics, "CREMA-D demographics")
    if _count_wavs(folder) >= EXPECTED_FILES["cremad"]:
        return
    tree_url = f"https://api.github.com/repos/{CREMAD_REPO}/git/trees/{CREMAD_REVISION}?recursive=1"
    tree = json.loads(_fetch(tree_url))["tree"]
    names = [
        entry["path"].split("/", 1)[1]
        for entry in tree
        if entry["path"].startswith("AudioWAV/") and entry["path"].endswith(".wav")
    ]
    media = f"https://media.githubusercontent.com/media/{CREMAD_REPO}/{CREMAD_REVISION}/AudioWAV"
    folder.mkdir(parents=True, exist_ok=True)

    def fetch_one(name: str) -> None:
        target = folder / name
        if not target.exists():
            partial = target.with_suffix(".part")
            partial.write_bytes(_fetch(f"{media}/{name}"))
            partial.replace(target)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(tqdm(pool.map(fetch_one, names), total=len(names), desc="CREMA-D", leave=False))


DOWNLOADERS: dict[str, Callable[[Path], None]] = {
    "cremad": download_cremad,
    "ravdess": download_ravdess,
    "tess": download_tess,
    "savee": download_savee,
}


def download_all(raw_dir: str | Path, corpora: tuple[str, ...]) -> dict[str, int]:
    """Download (or verify) each corpus; returns the number of ``.wav`` files found per corpus."""
    raw_dir = Path(raw_dir)
    counts: dict[str, int] = {}
    for corpus in corpora:
        logger.info("Fetching %s ...", corpus)
        DOWNLOADERS[corpus](raw_dir)
        counts[corpus] = _count_wavs(raw_dir / CORPUS_LAYOUT[corpus])
        status = "ok" if counts[corpus] == EXPECTED_FILES[corpus] else "UNEXPECTED COUNT"
        logger.info("%s: %d files (%s)", corpus, counts[corpus], status)
    return counts
