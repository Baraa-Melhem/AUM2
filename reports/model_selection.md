# Model selection

Every choice below was made on **dev out-of-fold (OOF) UAR**: the 96 dev speakers in five
speaker-disjoint folds. The selection tool, `tune.py`, never loads, predicts or reports the 25
held-out test speakers. Rows are sorted best first; the chosen setting is in **bold**.
Hyper-parameters not listed are those of [`configs/base.yaml`](../configs/base.yaml) and the
model's own config.

## WavLM-Large layer-weighted head (`configs/wavlm.yaml`)

The untuned first run (40 epochs, dropout 0.3, input dropout 0.1) reached OOF UAR 0.7573, and its
validation loss started rising after about 3 epochs: the head over-fits frozen features quickly.

### Round 1: epochs × dropout × input dropout

Run on CPU while the GPU trained the CRNN, and stopped after 8 of 12 combinations (the 40-epoch row
is the untuned run).

| train.epochs | model.ssl.dropout | model.ssl.input_dropout | OOF UAR | OOF WAR | OOF macro-F1 |
|---:|---:|---:|---:|---:|---:|
| **10** | **0.5** | **0.3** | **0.7696** | 0.7700 | 0.7720 |
| 10 | 0.5 | 0.1 | 0.7695 | 0.7701 | 0.7718 |
| 20 | 0.5 | 0.3 | 0.7673 | 0.7678 | 0.7692 |
| 10 | 0.3 | 0.3 | 0.7658 | 0.7664 | 0.7682 |
| 10 | 0.3 | 0.1 | 0.7656 | 0.7661 | 0.7678 |
| 20 | 0.5 | 0.1 | 0.7645 | 0.7651 | 0.7663 |
| 20 | 0.3 | 0.3 | 0.7627 | 0.7633 | 0.7642 |
| 20 | 0.3 | 0.1 | 0.7611 | 0.7616 | 0.7621 |
| 40 | 0.3 | 0.1 | 0.7573 | 0.7577 | 0.7577 |

### Round 2: epochs × dropout × learning rate (input dropout 0.1)

| train.epochs | model.ssl.dropout | train.learning_rate | OOF UAR | OOF WAR | OOF macro-F1 |
|---:|---:|---:|---:|---:|---:|
| **10** | **0.5** | **0.001** | **0.7695** | 0.7701 | 0.7718 |
| 14 | 0.5 | 0.001 | 0.7675 | 0.7679 | 0.7696 |
| 10 | 0.5 | 0.0005 | 0.7665 | 0.7670 | 0.7687 |
| 14 | 0.7 | 0.001 | 0.7659 | 0.7665 | 0.7684 |
| 6 | 0.5 | 0.001 | 0.7659 | 0.7663 | 0.7685 |
| 10 | 0.7 | 0.001 | 0.7657 | 0.7661 | 0.7684 |
| 14 | 0.5 | 0.0005 | 0.7651 | 0.7655 | 0.7673 |
| 14 | 0.7 | 0.0005 | 0.7650 | 0.7654 | 0.7674 |
| 10 | 0.7 | 0.0005 | 0.7619 | 0.7622 | 0.7642 |
| 6 | 0.5 | 0.0005 | 0.7596 | 0.7598 | 0.7619 |
| 6 | 0.7 | 0.001 | 0.7519 | 0.7520 | 0.7539 |
| 6 | 0.7 | 0.0005 | 0.7469 | 0.7469 | 0.7488 |

Both rounds agree on 10 epochs, dropout 0.5 and learning rate 1e-3. Input dropout 0.3 and 0.1
tie (0.7696 vs 0.7695); the nominal best, 0.3, is used. The final `train.py` run with these settings
reproduces OOF UAR 0.7696 exactly. **Gain from tuning: +0.012 OOF UAR.** HuBERT-Large
(`configs/hubert.yaml`) inherits these head settings unchanged.

## Hand-crafted functionals (`configs/handcrafted.yaml`)

### Logistic regression: inverse regularisation C

| model.handcrafted.logreg_c | OOF UAR | OOF WAR | OOF macro-F1 |
|---:|---:|---:|---:|
| **0.002** | **0.5112** | 0.5124 | 0.5080 |
| 0.001 | 0.5096 | 0.5106 | 0.5055 |
| 0.005 | 0.5095 | 0.5108 | 0.5070 |
| 0.02 | 0.5008 | 0.5020 | 0.4997 |
| 0.0005 | 0.4993 | 0.5002 | 0.4936 |

An earlier coarse sweep had C = 0.05 at 0.4858 and C = 0.2 at 0.4672.

### LightGBM: leaves per tree (800 trees, learning rate 0.03)

| model.handcrafted.lgbm_num_leaves | OOF UAR | OOF WAR | OOF macro-F1 |
|---:|---:|---:|---:|
| **31** | **0.5355** | 0.5363 | 0.5322 |
| 15 | 0.5288 | 0.5294 | 0.5259 |

LightGBM beats the best logistic regression by 0.024 OOF UAR and is the reported classical baseline.

## Log-mel CRNN (`configs/crnn.yaml`)

Not grid-searched: one 5-fold run costs about 1.5 h on the RTX 3050. Its settings are standard
choices for SER from log-mels (3 s crops, SpecAugment, tempo and VTLP perturbation, label smoothing
0.1, AdamW with weight decay 0.05, cosine schedule, EMA), fixed before any result was seen. After
a 2-epoch smoke test, and for speed only, time pooling was raised from ×4 to ×8 and training
crops shortened from 4 s to 3 s (26 → 17 s per epoch) before the full run.

## Ensemble composition and weights

`evaluate.py --ensemble ... --tune-weights` searches mixing weights on a simplex grid (step 0.05
for two members, 0.1 otherwise) and scores each by dev OOF UAR; the winning weights are then frozen
and applied to the test predictions. The members themselves were also chosen on dev OOF only:

| Candidate | Tuned weights | OOF UAR (tuned) | OOF UAR (equal weights) |
|---|---|---:|---:|
| WavLM-Large head | 1.0 | 0.7696 | 0.7696 |
| HuBERT-Large head | 1.0 | 0.7736 | 0.7736 |
| **WavLM + HuBERT** | **0.45 / 0.55** | **0.7923** | **0.7922** |
| WavLM + HuBERT + CRNN | 0.5 / 0.4 / 0.1 | 0.7931 | 0.7883 |
| WavLM + HuBERT + CRNN + functionals | 0.5 / 0.4 / 0.1 / 0.0 | 0.7931 | 0.7804 |
| WavLM + CRNN | 0.65 / 0.35 | 0.7730 | 0.7569 |

Combining two differently pre-trained SSL backbones is the one clear gain (+0.019 over the best
single model, almost independent of the weights). Adding the CRNN gains 0.0008, within noise and
only with tuned weights, so the reported ensemble is **WavLM + HuBERT**; the functionals get
weight 0.
