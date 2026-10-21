import numpy as np
import pandas as pd
import pytest

from aum2.config import SplitConfig
from aum2.data.splits import TEST_FOLD, assign_folds, check_speaker_disjoint


@pytest.fixture()
def manifest() -> pd.DataFrame:
    """A corpus-shaped manifest: 40/12/4/2 speakers, every speaker says every emotion 3 times."""
    rows = []
    layout = {
        "cremad": (40, ("male", "female")),
        "ravdess": (12, ("male", "female")),
        "savee": (4, ("male",)),
        "tess": (2, ("female",)),
    }
    emotions = ("angry", "disgust", "fear", "happy", "neutral", "sad")
    for corpus, (n, genders) in layout.items():
        for s in range(n):
            for e in emotions:
                for r in range(3):
                    rows.append(
                        {
                            "utt_id": f"{corpus}_{s}_{e}_{r}",
                            "corpus": corpus,
                            "speaker": f"{corpus}_{s}",
                            "gender": genders[s % len(genders)],
                            "emotion": e,
                        }
                    )
    return pd.DataFrame(rows)


def test_speaker_protocol_is_disjoint_and_covers_every_corpus(manifest):
    folds = assign_folds(manifest, SplitConfig())
    check_speaker_disjoint(manifest, folds)
    test = manifest[folds == TEST_FOLD]
    assert set(test["corpus"]) == {"cremad", "ravdess", "savee", "tess"}
    assert test.groupby("corpus")["speaker"].nunique().to_dict() == {
        "cremad": 8,
        "ravdess": 2,
        "savee": 1,
        "tess": 1,
    }
    # Gender-stratified: CREMA-D's 8 test speakers are 4 + 4.
    assert test[test.corpus == "cremad"].groupby("gender")["speaker"].nunique().tolist() == [4, 4]
    assert set(np.unique(folds)) == {TEST_FOLD, 0, 1, 2, 3, 4}


def test_split_is_deterministic_and_seed_dependent(manifest):
    a = assign_folds(manifest, SplitConfig(seed=7))
    assert np.array_equal(a, assign_folds(manifest, SplitConfig(seed=7)))
    assert not np.array_equal(a, assign_folds(manifest, SplitConfig(seed=8)))


def test_random_protocol_leaks_speakers(manifest):
    """The 2023 protocol: the same speaker lands in both test and dev."""
    folds = assign_folds(manifest, SplitConfig(protocol="random"))
    with pytest.raises(AssertionError):
        check_speaker_disjoint(manifest, folds)
