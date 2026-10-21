# Technical audit of the 2023 AUM2 project

This audit covers both sources of the original project:

- **Kaggle notebook** [`baraamelhem/audio-emotional-classification`](https://www.kaggle.com/code/baraamelhem/audio-emotional-classification)
  (40 cells, TensorFlow/Keras, inputs `uldisvalainis/audio-emotions` and `dmitrybabko/speech-emotion-recognition-en`).
- **GitHub repository** [`Baraa-Melhem/AUM2`](https://github.com/Baraa-Melhem/AUM2) at `de3bd02`
  (`AUM2.py`, five Keras `.h5` models of 24 MB each, a link to a sentiment model on Google Drive).

The verdict up front: **none of the reported numbers measure generalisation.** The 95.5-100 %
"test" accuracies are accuracies on training clips, and the 61-90 % validation accuracies are
inflated by duplicated clips that sit on both sides of a random split. The deployed script
then feeds the models an input that looks nothing like their training data. The sections below
prove each claim from the code; [`README.md`](README.md#results) gives the honest numbers.

## 1. What the original system does

```
audio-emotions (12 798 clips)  +  CREMA-D again  +  SAVEE again
        │  librosa.load(sr=None)  →  time_stretch(rate=1)  →  power mel (128 × T)
        │  tf.image.resize → 128 × 128      (no log, no normalisation)
        ▼
 random 75/25 split per model   →   5 tiny CNNs (Conv16 → MaxPool → Dense32 → softmax)
        │
        ▼
 hard-routed cascade:  M1 negative/positive → M2 {ang, dis, fea, sad} / M3 {hap, neu}
                       → M5 {ang, hap},  M4 {sad, neu}
        │
        ▼
 "test" on 200 TESS clips per emotion  →  95.5-100 % per emotion
```

`AUM2.py` wraps the same five models in a voice assistant: it records 7 s chunks, sends them to
the Google Web Speech API until it hears a wake phrase, records 20 s, predicts an emotion from
the recording, and runs a separate tweet-sentiment Keras model on the transcript.

## 2. Data-integrity defects (these invalidate the results)

| # | Defect | Evidence | Effect |
|---|--------|----------|--------|
| D1 | **The test set is part of the training set.** | `audio-emotions` is documented as CREMA-D 7 442 + TESS 2 800 + RAVDESS 2 076 + SAVEE 480. The test cell globs `Tess/YAF_angry`, `Tess/OAF_neutral`, … from the second dataset: the same TESS recordings. | The 95.5-100 % table is training accuracy. |
| D2 | **CREMA-D and SAVEE are added twice.** | Cells 7 and 9 append every CREMA-D and SAVEE file from `speech-emotion-recognition-en` on top of `audio-emotions`, which already holds them. | 7 922 exact duplicates. A random split puts most duplicate pairs on both sides, so the validation accuracy partly measures memorisation. |
| D3 | **Every SAVEE clip is labelled "angry" in the second copy.** | `target_emotion1 = "a"` and the test is `"a" in filename`, where `filename` is the full path `/kaggle/input/...`. "kaggle" contains an `a`, so the first branch always fires. | All 480 SAVEE clips, including the 60 *surprise* clips, are added as Angry. 420 of them contradict their correctly labelled twin from D2. |
| D4 | **Augmentation happens before the split.** | Happy clips get a `time_stretch(rate=1.4/1.3)` copy, and Neutral clips get an *identical* second copy (`rate=1` twice) before `train_test_split`. | More near-duplicate leakage. The class balance changes silently, and the "augmentation" adds no new information. |
| D5 | **No speaker independence.** | `train_test_split(..., test_size=0.25)` on clips. CREMA-D has 91 actors, RAVDESS 24, SAVEE 4, TESS 2. | The models can recognise the *speaker and sentence* instead of the emotion. Measured here, this alone inflates UAR by 7-13 points (§7). |
| D6 | **The validation set is also the test set.** | `model.fit(..., validation_data=(X_test, y_test))`, then `model.evaluate(X_test, y_test)`. | Nothing is left out for an unbiased estimate. |
| D7 | **Each sub-model is scored in isolation.** | Models 2-5 are trained and scored on clips that are already routed correctly. The cascade as a whole is only ever scored on the leaked TESS clips. | Routing errors compound, but no reported number includes them. |
| D8 | **The test accuracy uses a hard-coded denominator.** | `happy_acc = counter / 200` for every emotion. | It is only correct while each folder holds exactly 200 files. |

## 3. Signal-processing defects

| # | Defect | Effect |
|---|--------|--------|
| S1 | `librosa.load(sr=None)` keeps the native sample rates: CREMA-D 16 kHz, TESS 24.4 kHz, SAVEE 44.1 kHz, RAVDESS 48 kHz, live mic 44.1 kHz. | The mel axis spans 0-8 kHz for one corpus and 0-24 kHz for another. A hop of 512 is 32 ms at 16 kHz and 10.7 ms at 48 kHz. The network can identify the **corpus** from the spectral envelope, and corpus correlates with the label (D1-D3), which is a shortcut. |
| S2 | `tf.image.resize(..., (128, 128))` squeezes every clip to 128 frames whatever its length. | A 1.5 s TESS word and a 5 s RAVDESS sentence get the same width: speaking rate, a core emotion cue, is destroyed. In `AUM2.py` a **20 s** recording is squeezed into 128 frames, 10× more compression than anything seen in training, so live predictions are close to arbitrary. |
| S3 | A power mel spectrogram goes in with no log compression and no normalisation. | The dynamic range is about 10⁸, so a few loud bins dominate the first convolution. Per-corpus recording gain becomes another shortcut. |
| S4 | `time_stretch(rate=1)` runs on every clip. | A phase-vocoder pass that only adds artefacts, and it accounts for most of the 35-minute preprocessing time. |
| S5 | Leading and trailing silence are never trimmed (RAVDESS clips carry about 1 s of it). | With S2, silence takes a varying share of the 128 frames. |
| S6 | `AUM2.py` records stereo 44.1 kHz audio, writes it to disk, and reads it back. | It works because librosa downmixes, but it is wasted I/O, and it is the input mismatch behind S1 and S2. |

## 4. Modelling defects

| # | Defect | Effect |
|---|--------|--------|
| M1 | One Conv2D(16) layer, then `Flatten` into Dense(32): 63 504 × 32 = 2.03 M of the model's 2.03 M weights sit in a single dense layer. | The model is location-specific rather than translation-invariant along time, has almost no depth for spectro-temporal patterns, and overfits (train 92 % vs val 61-87 %). |
| M2 | No BatchNorm, dropout, weight decay, LR schedule, early stopping or checkpointing. | The last epoch is kept whatever its validation score (model 3 val loss jumps to 1.03 at epoch 7). |
| M3 | The hard-routed cascade of five models has inconsistent semantics. | Errors compound with no recovery. The "positive" branch can output Angry or Sad, and its first-level confidence is discarded. The printed "percentage" is the *last* sub-model's softmax, not the probability of the predicted emotion. |
| M4 | `binary_crossentropy` on a 2-unit softmax. | It works, but it is the wrong loss for the output layer. |
| M5 | No seeds anywhere (`random.shuffle` is unseeded). | Not reproducible. |
| M6 | Evaluation is accuracy only. | No UAR, F1, confusion matrix or per-corpus breakdown. Accuracy hides class-specific failure: Disgust and Fear are the hard classes in SER. |

## 5. Software-engineering defects

| # | Defect |
|---|--------|
| E1 | Hard-coded absolute Windows paths (`C:\Users\User\Desktop\...`) and Kaggle paths. Nothing runs on another machine. |
| E2 | The notebook saves `.pkl` files (`joblib.dump` of Keras models, which is fragile and Python-version dependent) while the repo ships `.h5` files: two artefact formats, and nothing links the `.h5` files to a training run. |
| E3 | The same 8-line preprocessing block is copy-pasted ~20 times, and the five model definitions are copy-pasted too. |
| E4 | All state is global (`model`, `model2`, …, `X_train` rebound five times). `predict_model` shadows the `speech_recognition` module alias `sr` locally. |
| E5 | `NLP_on_text` builds a POS-tagged string (`newcleantext`) and then lemmatises the *original* text object instead. The POS tags never reach the lemmatiser, and spaCy and NLTK are imported inside the function on every call. |
| E6 | The sentiment model and tokenizer live only on Google Drive, with no training code. The pipeline cannot be reproduced from the repository. |
| E7 | The live loop sends every 7 s of microphone audio to Google's cloud ASR (a privacy issue), loops forever, and matches the wake word against 11 hand-listed mis-transcriptions. |
| E8 | No `requirements.txt`, tests, config, CLI, logging or docs. The README is two lines. `keras.preprocessing.sequence.pad_sequences` is gone in Keras 3. |
| E9 | 120 MB of opaque binaries are committed to git. |

## 6. Where accuracy was left on the table

1. **Representation.** Self-supervised speech models (WavLM, HuBERT, wav2vec 2.0) encode
   prosody and voice quality far better than a 16-filter CNN trained from scratch on
   ~10 k clips. They are the established state of the art for SER.
2. **Input fidelity.** Log compression, one sample rate, silence trimming, and variable-length
   inputs with masking instead of resizing.
3. **Architecture.** Deep residual convolutions, temporal modelling (BiGRU), and attentive
   statistics pooling over time instead of `Flatten`.
4. **Regularisation and augmentation,** applied to training batches only: SpecAugment, tempo and
   vocal-tract-length perturbation, additive noise, label smoothing, weight decay, EMA weights.
5. **One calibrated 6-way classifier** instead of a hard cascade, with **fold ensembles** on top.

See [README.md § Methodology](README.md#methodology) for what was built in response to each
finding.

## 7. Measured outcome

Every number below is UAR (mean per-class recall, chance = 0.167) on the 25 held-out test speakers,
from the runs in [`reports/runs/`](reports/runs/).

| Question | Answer |
|---|---|
| How good is the 2023 method when nothing leaks? | **0.402** [0.363, 0.424]: the same features, five CNNs, cascade and training recipe, re-implemented faithfully (`configs/legacy.yaml`). |
| How much does the 2023 *protocol* alone inflate it? | Under a random utterance-level split the same method scores 0.534 (+13 points) and the best model 0.877 (+7 points), with TESS at 1.000. The duplicated and test-in-train data (D1–D3) come on top of that. |
| What does a sound log-mel model reach? | 0.729 for the CRNN (2.2 M parameters, trained from scratch). |
| What does the state of the art reach? | 0.798 for the selected frozen WavLM-Large + HuBERT-Large ensemble, 0.805 for WavLM alone; the difference is not significant (p = 0.42). |
| Did the source data have problems of its own? | Yes: three CREMA-D pairs are byte-identical files with different labels. They are now dropped. |
