"""
predict_wrapper.py — TrustECG STGNN+RR, single entry point for the ensemble
===========================================================================
    from predict_wrapper import predict_proba
    probs = predict_proba(x)      # x: (N, 5000, 12) -> (N, 5) float64

Class order is [SB, AFIB, SR, ST, SVT]. Rows sum to 1.

Note on "do not re-filter"
--------------------------
STGNN_RR does not consume raw signal. It consumes beat windows
(n_beats, 400, 12) plus R-R features (n_beats, 6), so this wrapper MUST run
the training-time pipeline (bandpass -> notch -> wavelet -> quality screen ->
Pan-Tompkins -> beat windowing -> per-lead normalisation -> R-R features).
That is not a bug and it is not double-preprocessing: normalize_beats
recomputes statistics on the FILTERED recording exactly as training did, so
pre-z-scored input lands on the same scale either way. Skipping this step
would hand the model an input format it has never seen.

Files this needs, all resolved relative to this file unless overridden with
the TRUSTECG_CKPT_DIR environment variable:

    ./model_stgnn_rr.py, ./model_stgnn.py, ./dataset_rr.py,
    ./config_physionet.py, ./preprocessing_physionet_stgnn.py
    <ckpt_dir>/stgnn_rr_best.pt
    <ckpt_dir>/calibration_rr_epoch7.json

Default <ckpt_dir> is ../outputs/checkpoints_stgnn_rr relative to this file.
Set TRUSTECG_CKPT_DIR to put the two files anywhere else.

Runs on CPU or GPU; picks CUDA when available, CPU otherwise. The model is
built once on first call and cached, so repeated calls pay only inference.
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config_physionet import PhysioNetConfig as PNConfig          # noqa: E402
from dataset_rr import compute_rr_features                        # noqa: E402
from model_stgnn_rr import STGNN_RR                               # noqa: E402
from preprocessing_physionet_stgnn import (                       # noqa: E402
    FS,
    LEAD_II_IDX,
    LEAD_V1_IDX,
    MIN_BEATS_PER_RECORD,
    MIN_VALID_LEADS,
    POST_R_SAMPLES,
    PRE_R_SAMPLES,
    USE_WAVELET_DENOISE,
    bandpass_filter,
    cross_validate_r_peaks,
    detect_flatline_leads,
    detect_r_peaks,
    detect_step_artifacts,
    extract_beats,
    normalize_beats,
    notch_filter,
    wavelet_denoise,
)

CLASS_ORDER = ["SB", "AFIB", "SR", "ST", "SVT"]

_CKPT_DIR = os.environ.get(
    "TRUSTECG_CKPT_DIR",
    os.path.join(os.path.dirname(_HERE), "outputs", "checkpoints_stgnn_rr"),
)

_STATE = {}   # lazily populated: model, device, calibration a/b


# ---------------------------------------------------------------------------
# Model loading — once per process
# ---------------------------------------------------------------------------

def _load():
    if _STATE:
        return _STATE

    PNConfig.load_label_vocab()
    if list(PNConfig.CLASS_NAMES) != CLASS_ORDER:
        raise RuntimeError(f"class order mismatch: checkpoint has "
                           f"{PNConfig.CLASS_NAMES}, ensemble wants {CLASS_ORDER}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(_CKPT_DIR, "stgnn_rr_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Put stgnn_rr_best.pt and "
            f"calibration_rr_epoch7.json there, or set TRUSTECG_CKPT_DIR.")

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    tc = ck.get("train_config", {})
    epoch = ck.get("epoch", 0)

    cal_path = os.path.join(_CKPT_DIR, f"calibration_rr_epoch{epoch}.json")
    if not os.path.exists(cal_path):
        # Uncalibrated this model scores 0.899 instead of 0.961. Failing loudly
        # beats silently contributing the weaker model to the ensemble.
        raise FileNotFoundError(
            f"Calibration missing at {cal_path}. Refusing to predict "
            f"uncalibrated — the saved probs were calibrated and would not match.")
    with open(cal_path) as fh:
        cj = json.load(fh)

    model = STGNN_RR(num_classes=len(CLASS_ORDER),
                     node_dim=tc.get("node_dim", 128),
                     gat_heads=tc.get("gat_heads", 4),
                     gru_hidden=tc.get("gru_hidden", 128),
                     dropout=tc.get("dropout", 0.1)).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    _STATE.update(
        model=model,
        device=device,
        a=np.asarray(cj["vector_scale_a"], dtype=np.float32),
        b=np.asarray(cj["vector_bias_b"], dtype=np.float32),
    )
    return _STATE


# ---------------------------------------------------------------------------
# Per-record preprocessing — relaxed, never rejects a row
# ---------------------------------------------------------------------------

def _preprocess_one(sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """sig: (5000, 12) float32 -> (beats, rr, lead_valid, status)"""
    status: List[str] = []

    filtered = bandpass_filter(sig.astype(np.float32))
    filtered = notch_filter(filtered)
    if USE_WAVELET_DENOISE:
        filtered = wavelet_denoise(filtered)

    lead_valid = ~(detect_flatline_leads(filtered) | detect_step_artifacts(filtered))

    if not lead_valid[LEAD_II_IDX]:
        lead_valid[LEAD_II_IDX] = True          # needed for R-peak detection
        status.append("lead_II_forced")
    if int(lead_valid.sum()) < MIN_VALID_LEADS:
        status.append(f"few_valid_leads({int(lead_valid.sum())})")

    peaks, _ = detect_r_peaks(filtered[:, LEAD_II_IDX])
    if lead_valid[LEAD_V1_IDX] and len(peaks) >= MIN_BEATS_PER_RECORD:
        peaks_v1, _ = detect_r_peaks(filtered[:, LEAD_V1_IDX])
        validated, _ = cross_validate_r_peaks(peaks, peaks_v1)
        if len(validated) >= MIN_BEATS_PER_RECORD:
            peaks = validated

    beats, kept = extract_beats(filtered, peaks)

    if beats.shape[0] < MIN_BEATS_PER_RECORD:
        # Evenly spaced pseudo-beats at ~75 bpm. Morphology still sees real
        # signal; only the R-R features go meaningless, so the row is flagged.
        status.append("synthetic_beats")
        stride = int(round(0.8 * FS))
        pseudo = np.arange(PRE_R_SAMPLES, sig.shape[0] - POST_R_SAMPLES, stride, dtype=int)
        if len(pseudo) < MIN_BEATS_PER_RECORD:
            pseudo = np.linspace(PRE_R_SAMPLES, sig.shape[0] - POST_R_SAMPLES - 1,
                                 MIN_BEATS_PER_RECORD).astype(int)
        beats, kept = extract_beats(filtered, pseudo)

    beats = normalize_beats(beats, filtered)
    beats = np.nan_to_num(beats, nan=0.0, posinf=0.0, neginf=0.0)
    rr = compute_rr_features(kept, n_beats=beats.shape[0])
    return beats, rr, lead_valid, status


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_proba(x: np.ndarray) -> np.ndarray:
    """
    x   : (N, 5000, 12) float32 array — 10 s at 500 Hz, time on axis 1,
          leads on axis 2. A single (5000, 12) record is also accepted.
    out : (N, 5) float64 softmax probabilities. Rows sum to 1, class order
          [SB, AFIB, SR, ST, SVT].

    Any N from 1 to 64 (or larger) works. Records are preprocessed one at a
    time because R-peak detection and beat count are per-record; the model
    itself stays loaded across calls.
    """
    st = _load()
    model, device, a, b = st["model"], st["device"], st["a"], st["b"]

    arr = np.asarray(x)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[1:] != (5000, 12):
        raise ValueError(f"expected (N, 5000, 12), got {arr.shape}")

    logits = np.zeros((arr.shape[0], len(CLASS_ORDER)), dtype=np.float32)
    for i in range(arr.shape[0]):
        beats, rr, lead_valid, _ = _preprocess_one(np.asarray(arr[i]))
        z = model(torch.from_numpy(beats).float().to(device),
                  torch.from_numpy(rr).float().to(device),
                  torch.from_numpy(lead_valid).bool().to(device))
        logits[i] = z.float().cpu().numpy()

    cal = logits * a + b
    e = np.exp(cal - cal.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float64)


def predict_status(x: np.ndarray) -> List[str]:
    """Per-row preprocessing status, same order as predict_proba's rows.

    'ok', or a '|'-joined subset of lead_II_forced / few_valid_leads(k) /
    synthetic_beats. Rows that are not 'ok' are degraded and less trustworthy.
    """
    arr = np.asarray(x)
    if arr.ndim == 2:
        arr = arr[None, ...]
    out = []
    for i in range(arr.shape[0]):
        _, _, _, status = _preprocess_one(np.asarray(arr[i]))
        out.append("|".join(status) if status else "ok")
    return out
