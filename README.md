# TrustECG — Explainable 12-Lead ECG Rhythm Classification

An ensemble of three deep-learning models that classifies a 10-second 12-lead ECG into one of
five rhythms, and explains which parts of the signal the decision rested on.

> **Research prototype — not a medical device.** Nothing here is a diagnosis. The models have
> not been clinically validated and must not be used for any medical decision.

## Rhythms

`SB` Sinus Bradycardia · `AFIB` Atrial Fibrillation · `SR` Sinus Rhythm ·
`ST` Sinus Tachycardia · `SVT` Supraventricular Tachycardia

## Results

Held-out test set, 6,363 recordings. The combination method was chosen by 5-fold
cross-validation **on the validation set only**; the test set was used once, for the chosen
method.

| | Macro F1 | Accuracy | Calibration error (ECE) |
|---|---|---|---|
| CNN-BiLSTM alone | 95.18% | 97.66% | 0.128 |
| InceptionTime alone | 93.38% | 95.95% | 0.010 |
| STGNN+RR alone | 88.37% | 95.14% | 0.007 |
| Equal-weight average | 95.52% | 97.88% | 0.070 |
| **Stacked ensemble (final)** | **96.25%** | **98.40%** | **0.003** |

The stacked ensemble beats the best single model significantly (McNemar p < 0.001; bootstrap
95% CI of the macro-F1 difference +0.14 to +1.94 points) and is close to perfectly calibrated:
a stated confidence of 90% really is right about 90% of the time.

Per-class F1: SB 0.994 · AFIB 0.981 · SR 0.978 · ST 0.981 · **SVT 0.877 (weakest)**

The stacker relies about 50% on CNN-BiLSTM, 28% on InceptionTime and 22% on STGNN+RR.

### Are the explanations trustworthy?

Tested by deletion on 40 recordings. Blanking the regions the system calls important costs
**3.1×** more confidence than blanking random regions, and **12×** more than blanking the
regions it calls unimportant (AOPC 0.175 / 0.057 / 0.014). That ordering held on 75% of
recordings individually — informative on average, not guaranteed for any single recording.

## Install

```bash
git clone https://github.com/<you>/<repo>.git
cd <repo>
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

### Model weights

Checkpoints are too large for the repository. Download them from the
**[Releases](../../releases)** page and place them as:

```
outputs/checkpoints_physionet/physionet_cnn_bilstm2_best_model.pth
teammates/inception/best_inception_5class_PROTECTED.pth
teammates/stgnn/outputs/checkpoints_stgnn_rr/stgnn_rr_best.pt
teammates/stgnn/outputs/checkpoints_stgnn_rr/calibration_rr_epoch7.json
```

### Data

Trained on the PhysioNet/CinC 2021 Chapman-Shaoxing and Ningbo 12-lead ECG databases.
Download from [physionet.org](https://physionet.org/) into `data/`, then run:

```bash
python utils/preprocessing_physionet.py
```

## Analysing one recording

```bash
python analyze_ecg.py --input path/to/recording
```

Accepts a WFDB record path (no extension), a `.csv` of raw 12-lead signal, or a preprocessed
`.npy` of shape (5000, 12). Writes a PDF report, a `results.json` and the explanation figures
to `outputs/web_analysis/<name>/`.

```
  prediction : AFIB (Atrial Fibrillation)  confidence 96.8%
  models     : 3/3 agree -> cnn_bilstm=AFIB, inception=AFIB, stgnn=AFIB
  top leads  : V4, V5, V6
```

## Reproducing the published numbers

The saved per-model probabilities are committed, so steps 1–2 run without the checkpoints.

```bash
python ensemble_improve.py        # compares 10 combination methods, picks one on validation
python ensemble_combiner.py       # fits and saves the chosen ensemble -> 0.9625
python -m ensemble_xai.adapters   # checks all three models reproduce their saved probabilities
python -m ensemble_xai.deletion_check --n_per_class 8   # faithfulness test
python generate_ensemble_xai.py   # explanation figures, one recording per rhythm
```

## Layout

```
analyze_ecg.py            analyse one recording end to end (prediction + XAI + PDF)
ensemble_improve.py       compare combination methods, choose one on validation only
ensemble_combiner.py      fit and save the final stacked ensemble
ensemble_xai/             model-agnostic occlusion explanations
  adapters.py               wraps all three models behind one predict_proba
  occlusion.py              the occlusion sweep
  deletion_check.py         quantitative faithfulness test
xai/                      Grad-CAM++ explanations for CNN-BiLSTM alone
models/, utils/           architectures, preprocessing, dataset, metrics
teammates/                the two partner models and their inference wrappers
```

## Limitations

- Research prototype; not clinically validated, not a medical device.
- One public database, one recording setup. Performance elsewhere is unknown and likely lower.
- Only five rhythms. Anything else is forced into one of them.
- SVT is the weakest class (F1 0.877) and the rarest in the data.
- Occlusion removes one region at a time, so information duplicated across leads can be
  under-credited. Regions arguing *against* a prediction are shown as zero, not negative.
- STGNN+RR detects individual heartbeats before classifying, so perturbing the signal can
  change the segmentation itself, adding noise to its share of the explanation.
- A recording is 10 seconds; intermittent rhythms may simply not appear in it.

## Credits

| Component | Author |
|---|---|
| CNN-BiLSTM, ensemble, explainability, integration | AVANI |
| InceptionTime | ANANYA |
| STGNN+RR | ALMA |

## License

