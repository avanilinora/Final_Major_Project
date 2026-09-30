# TrustECG STGNN+RR — ensemble handoff

Model: spatio-temporal GNN (ResNet1D per lead → GAT over a 12-lead graph →
BiGRU → temporal attention), with per-class vector calibration applied before
the softmax. Checkpoint is epoch 7, the same one that produced the
`probs_val.npy` / `probs_test.npy` already sent. Nothing was retrained.

## 1. Row alignment and accuracy against your labels

Checked with your `y_test.npy` / `y_val.npy` against the probs already sent:

| Split | n | Accuracy | Macro F1 | Macro AUROC |
|---|---|---|---|---|
| test | 6,363 | **0.9514** | **0.8837** | 0.9912 |
| val  | 6,362 | 0.9450 | 0.8581 | 0.9889 |

Rows align and class order is `[SB, AFIB, SR, ST, SVT]` as required.

**On the 0.9611 vs 0.9514 gap** — this is not a row-ordering problem. A
misaligned ordering would collapse accuracy to chance (~0.2), not cost one
point. 0.9611 is our standalone number on *our* split, which applies a quality
screen that drops 22.4% of the database. Your arrays keep those recordings, so
1,453 of the 6,363 test rows are degraded and are scored anyway. The gap is
exactly that relaxed gate.

`status_test.csv` / `status_val.csv` list the per-row status
(`ok`, `lead_II_forced`, `few_valid_leads(k)`, `synthetic_beats`). Worth
down-weighting the non-`ok` rows if your ensemble supports it. Only 5 rows
across both splits are `synthetic_beats`, which are the least trustworthy.

## 2. Running the model

```python
import sys; sys.path.insert(0, "utils")
from predict_wrapper import predict_proba

probs = predict_proba(x)   # x: (N, 5000, 12) float32  ->  (N, 5) float64
```

Rows sum to 1, class order `[SB, AFIB, SR, ST, SVT]`. Any N works, including 1.
A bare `(5000, 12)` record is accepted too.

There is also `predict_status(x)`, which returns the same per-row status
strings as the CSVs, so you can tell whether a modified recording has fallen
into a degraded path.

### Layout

Keep the two folders as shipped, and the default paths resolve on their own:

```
utils/…                                    all .py files
outputs/checkpoints_stgnn_rr/stgnn_rr_best.pt
outputs/checkpoints_stgnn_rr/calibration_rr_epoch7.json
```

To put the checkpoint elsewhere, set `TRUSTECG_CKPT_DIR` to its folder.

You will see one warning on import:
`[PhysioNetConfig] No label_vocab.json … using default CLASS_NAMES`.
That is expected and harmless — the defaults are already the right class order.
The wrapper raises if the order is ever wrong, so it cannot pass silently.

### About "do not re-filter my input"

This one requirement can't be honoured, and it's deliberate. STGNN_RR does not
consume raw signal — it consumes beat windows `(n_beats, 400, 12)` plus R-R
features `(n_beats, 6)`. The wrapper has to run the training-time pipeline
(bandpass → notch → wavelet → quality screen → Pan-Tompkins R-peak detection →
beat windowing → per-lead normalisation → R-R features) to produce them.

Your pre-z-scoring is not undone or doubled: `normalize_beats` recomputes
statistics on the filtered recording, which is what training did, so the beats
land on the same scale either way. The self-test below reproduces the submitted
probs exactly, which confirms it.

For the explainability work this matters: perturbing the raw signal will
propagate through R-peak detection, so a large enough perturbation can change
the beat segmentation itself, not just the morphology. Use `predict_status` to
spot rows where that has happened.

### Speed

The model is built once at first call and cached, so repeated calls pay only
inference.

| | per record | 240 calls |
|---|---|---|
| GPU (CUDA) | ~16 ms | ~4 s |
| CPU only | ~49 ms | ~12 s |

Runs on either; it picks CUDA when available and falls back to CPU
automatically. No GPU needed. Batching gives no speedup — the cost is
per-record preprocessing, not the forward pass — so call it however is
convenient.

## 3. Self-test

`selftest.py` reproduces your check. Our output, on the first 64 rows of your
`X_test.npy` against the submitted `probs_test.npy`:

```
shape: (64, 5)  rows sum to 1: True
max abs diff vs saved probs: 0.0
argmax agreement: 1.0
batch  1: max diff vs batched 0.000e+00
batch  7: max diff vs batched 0.000e+00
batch 64: max diff vs batched 0.000e+00
```

Exact on GPU. On CPU the max diff is 3.7e-5, from float non-determinism across
devices, well inside your 1e-3 bar.

## Files

| File | |
|---|---|
| `utils/predict_wrapper.py` | the entry point — `predict_proba`, `predict_status` |
| `utils/model_stgnn_rr.py` | architecture (`STGNN_RR`) |
| `utils/model_stgnn.py`, `utils/dataset_rr.py`, `utils/dataset_stgnn.py` | imported by the above |
| `utils/preprocessing_physionet_stgnn.py` | the training-time signal pipeline |
| `utils/config_physionet.py`, `utils/calibrate_stgnn_rr.py` | config and calibration paths |
| `outputs/checkpoints_stgnn_rr/stgnn_rr_best.pt` | epoch-7 checkpoint |
| `outputs/checkpoints_stgnn_rr/calibration_rr_epoch7.json` | per-class scale + bias |
| `status_test.csv`, `status_val.csv` | per-row preprocessing status |
| `summary.json` | export metadata |
| `selftest.py` | the step-3 check |

Requires `numpy`, `torch`, `scipy`, `pywt` (PyWavelets), `neurokit2`.
