from pathlib import Path

import pytest

from aum2.data.corpora import (
    FilenameError,
    parse_cremad,
    parse_ravdess,
    parse_savee,
    parse_tess,
)
from aum2.data.dataset import build_manifest


def test_cremad_parser_reads_actor_sentence_emotion_and_sex():
    u = parse_cremad(Path("1091_IEO_FEA_HI.wav"), {"1091": "female"})
    assert (u.speaker, u.emotion, u.text_id, u.intensity, u.gender) == (
        "cremad_1091",
        "fear",
        "IEO",
        "HI",
        "female",
    )


def test_ravdess_parser_maps_codes_and_gender_parity():
    u = parse_ravdess(Path("03-01-05-02-01-02-12.wav"))
    assert (u.speaker, u.emotion, u.gender, u.intensity) == (
        "ravdess_12",
        "angry",
        "female",
        "strong",
    )
    assert parse_ravdess(Path("03-01-02-01-01-01-07.wav")).emotion == "calm"
    assert parse_ravdess(Path("03-02-03-01-01-01-01.wav")) is None  # song, not speech


def test_tess_parser_handles_surprise_code_and_case():
    assert parse_tess(Path("YAF_youth_ps.wav")).emotion == "surprise"
    u = parse_tess(Path("OAF_Fear_fear.wav"))
    assert (u.speaker, u.emotion, u.gender) == ("tess_OAF", "fear", "female")


@pytest.mark.parametrize("folder", ["/kaggle/input/speech-emotion-recognition-en/Savee", "C:/data"])
def test_savee_parser_ignores_the_directory(folder):
    """Regression for the 2023 bug: `"a" in full_path` labelled every SAVEE clip angry."""
    assert parse_savee(Path(folder) / "DC_n01.wav").emotion == "neutral"
    assert parse_savee(Path(folder) / "JK_sa15.wav").emotion == "sad"
    assert parse_savee(Path(folder) / "KL_su03.wav").emotion == "surprise"


@pytest.mark.parametrize(
    "parser, name",
    [
        (lambda p: parse_cremad(p, {}), "1001_DFA_XYZ_XX.wav"),
        (parse_ravdess, "03-01-05-01.wav"),
        (parse_tess, "OAF_back_bored.wav"),
        (parse_savee, "XX_a01.wav"),
        (parse_savee, "DC_a01.mp3"),
    ],
)
def test_parsers_reject_unexpected_names(parser, name):
    with pytest.raises(FilenameError):
        parser(Path(name))


def test_manifest_keeps_six_classes_and_namespaced_speakers(raw_dir):
    manifest = build_manifest(
        raw_dir,
        ("cremad", "ravdess", "tess", "savee"),
        ("angry", "disgust", "fear", "happy", "neutral", "sad"),
    )
    assert set(manifest["emotion"]) == {"angry", "disgust", "fear", "happy", "neutral", "sad"}
    assert manifest["utt_id"].is_unique
    assert manifest.groupby("corpus")["speaker"].nunique().to_dict() == {
        "cremad": 6,
        "ravdess": 4,
        "savee": 4,
        "tess": 2,
    }
    assert (
        manifest["label"]
        == manifest["emotion"].map(
            {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "neutral": 4, "sad": 5}
        )
    ).all()
    assert not manifest["path"].str.contains("03-02-").any()  # RAVDESS song file skipped
