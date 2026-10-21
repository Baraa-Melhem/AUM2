# Legacy (2023) code, for reference only

| File | What it is |
|---|---|
| [`AUM2.py`](AUM2.py) | The original voice-assistant script from this repository (commit `de3bd02`), verbatim. |
| [`kaggle_notebook.py`](kaggle_notebook.py) | The code cells of the [2023 Kaggle notebook](https://www.kaggle.com/code/baraamelhem/audio-emotional-classification), transcribed from the published version. |

Neither file is imported, tested or linted. The five 24 MB Keras models (`Model/audio models .h5/`)
and the Google Drive link to the sentiment model remain in the git history (commit `de3bd02`) but
were removed from the tree: they were trained with the leaky protocol described in
[`../AUDIT.md`](../AUDIT.md), and the sentiment model's training code was never published.

The original *method* (features, five CNNs, cascade routing, training recipe) is re-implemented
faithfully in [`src/aum2/models/legacy.py`](../src/aum2/models/legacy.py) and re-evaluated on
unseen speakers by `python train.py --config configs/legacy.yaml`.
