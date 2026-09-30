from __future__ import annotations

"""
utils/preprocessing_physionet_stgnn.py
=======================================
Beat-synchronized, multi-label, quality-screened preprocessing for the
Chapman-Shaoxing / Ningbo 12-lead ECG dataset, built for the STGNN pipeline:

    12-lead ECG -> filter -> quality screen -> R-peak detect -> beat window
    -> per-recording per-lead normalize -> (beats, multi-hot label) per record

This replaces the whole-recording / single-label design of
preprocessing_physionet.py. Key differences:

  * No fixed-length centre-crop. Each recording yields a variable number of
    beats, so output is one .npz per record (ragged, can't stack into one
    .npy) plus a JSON index for the splits.
  * Labels are multi-hot (assign_multi_label), loss should be BCE, not CE.
  * A signal-quality screen (flatline + step/lead-off artifact detection)
    runs before R-peak detection and can reject a recording or individual
    leads.
  * R-peaks are detected on lead II (primary: simplified Pan-Tompkins;
    fallback: wavelet-enhanced peak picking if too few beats are found) and
    cross-validated against lead V1 to drop spurious detections.
  * Beats are windowed 300 ms pre-R / 500 ms post-R (configurable),
    normalized using per-lead mean/std computed once over the full filtered
    recording.

CODE_MERGE_MAP (AF -> AFIB) and the SB/AFIB/SR/ST/SVT whitelist are carried
over unchanged from the previous script.

Usage
-----
    from utils.preprocessing_physionet_stgnn import run_preprocessing_pipeline
    run_preprocessing_pipeline()
"""

import glob
import json
import logging
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pywt
import wfdb
from scipy.signal import butter, filtfilt, iirnotch, find_peaks

from config_physionet import PhysioNetConfig as PNConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config, with fallbacks so this runs even if config_physionet.py hasn't been
# updated yet. Add these to PhysioNetConfig for real; the getattr defaults
# below are just so the script doesn't hard-crash in the meantime.
# ---------------------------------------------------------------------------
FS: int = PNConfig.SAMPLING_FREQ
NUM_LEADS: int = PNConfig.NUM_LEADS

PRE_R_MS: float = getattr(PNConfig, "PRE_R_MS", 300.0)
POST_R_MS: float = getattr(PNConfig, "POST_R_MS", 500.0)
PRE_R_SAMPLES: int = int(round(PRE_R_MS / 1000.0 * FS))
POST_R_SAMPLES: int = int(round(POST_R_MS / 1000.0 * FS))
WINDOW_LEN: int = PRE_R_SAMPLES + POST_R_SAMPLES

NOTCH_FREQ: float = getattr(PNConfig, "NOTCH_FREQ", 50.0)
NOTCH_Q: float = getattr(PNConfig, "NOTCH_Q", 30.0)
USE_WAVELET_DENOISE: bool = getattr(PNConfig, "USE_WAVELET_DENOISE", True)

# Standard 12-lead header order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
LEAD_II_IDX: int = getattr(PNConfig, "LEAD_II_IDX", 1)
LEAD_V1_IDX: int = getattr(PNConfig, "LEAD_V1_IDX", 6)

FLATLINE_STD_THRESH: float = getattr(PNConfig, "FLATLINE_STD_THRESH", 1e-3)  # mV
STEP_ARTIFACT_Z: float = getattr(PNConfig, "STEP_ARTIFACT_Z", 5.0)
STEP_ARTIFACT_WINDOW_MS: float = getattr(PNConfig, "STEP_ARTIFACT_WINDOW_MS", 1000.0)
MIN_VALID_LEADS: int = getattr(PNConfig, "MIN_VALID_LEADS", 10)

MIN_BEATS_PER_RECORD: int = getattr(PNConfig, "MIN_BEATS_PER_RECORD", 3)
R_PEAK_REFRACTORY_MS: float = getattr(PNConfig, "R_PEAK_REFRACTORY_MS", 250.0)
CROSS_VALIDATION_TOL_MS: float = getattr(PNConfig, "CROSS_VALIDATION_TOL_MS", 100.0)
CROSS_VALIDATION_MIN_AGREEMENT: float = getattr(
    PNConfig, "CROSS_VALIDATION_MIN_AGREEMENT", 0.5
)

BEATS_SUBDIR = "beats"

CODE_MERGE_MAP: Dict[str, str] = {
    "164890007": "164889003",  # AF -> AFIB
}
WHITELIST_CODES = [
    "426177001",  # SB
    "164889003",  # AFIB
    "426783006",  # SR
    "427084000",  # ST
    "426761007",  # SVT
]


def _normalize_dx_codes(dx_codes: List[str]) -> List[str]:
    return [CODE_MERGE_MAP.get(c, c) for c in dx_codes]


# ===========================================================================
# Record discovery / label vocab / Dx parsing — unchanged from the previous
# script, reproduced here so this file is drop-in standalone.
# ===========================================================================

def find_all_records(data_path: str = PNConfig.DATA_PATH) -> List[str]:
    records_root = os.path.join(data_path, PNConfig.RECORDS_SUBDIR)
    if not os.path.isdir(records_root):
        raise FileNotFoundError(f"Could not find {records_root}.")
    hea_files = glob.glob(os.path.join(records_root, "**", "*.hea"), recursive=True)
    record_paths = sorted(p[:-4] for p in hea_files)
    logger.info("Found %d records under %s", len(record_paths), records_root)
    return record_paths


def load_snomed_names(csv_path: str = PNConfig.CONDITION_NAMES_CSV) -> Dict[str, str]:
    import csv
    if not os.path.exists(csv_path):
        logger.warning("ConditionNames_SNOMED-CT.csv not found at %s", csv_path)
        return {}
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        code_col = next((c for c in fieldnames if "snomed" in c.lower()), fieldnames[0])
        name_col = next(
            (c for c in fieldnames if c != code_col and ("name" in c.lower() or "acronym" in c.lower())),
            fieldnames[1] if len(fieldnames) > 1 else code_col,
        )
        mapping: Dict[str, str] = {}
        for row in reader:
            code = str(row[code_col]).strip()
            name = str(row[name_col]).strip()
            if code:
                mapping[code] = name
    return mapping


def get_dx_codes(record_path: str) -> List[str]:
    header = wfdb.rdheader(record_path)
    comments = header.comments or []
    for line in comments:
        stripped = line.lstrip("#").strip()
        if stripped.lower().startswith("dx"):
            _, _, value = stripped.partition(":")
            return [c.strip() for c in value.split(",") if c.strip()]
    return []


def build_label_vocab(
    record_paths: List[str],
    snomed_names: Dict[str, str],
    include_codes: List[str] = WHITELIST_CODES,
) -> Tuple[Dict[str, int], List[str], List[str]]:
    counter: Counter = Counter()
    for rp in record_paths:
        try:
            codes = get_dx_codes(rp)
        except Exception as exc:
            logger.warning("Could not read header for %s: %s", rp, exc)
            continue
        codes = _normalize_dx_codes(codes)
        counter.update(codes)

    code_to_idx: Dict[str, int] = {}
    class_names: List[str] = []
    class_codes: List[str] = []
    for idx, code in enumerate(include_codes):
        code_to_idx[code] = idx
        class_codes.append(code)
        class_names.append(snomed_names.get(code, code))
        logger.info("  class %d : %-12s %-30s (n=%d)", idx, code,
                    snomed_names.get(code, "?"), counter.get(code, 0))
    return code_to_idx, class_names, class_codes


def assign_multi_label(dx_codes: List[str], code_to_idx: Dict[str, int]) -> Optional[np.ndarray]:
    dx_codes = _normalize_dx_codes(dx_codes)
    present = [code_to_idx[c] for c in dx_codes if c in code_to_idx]
    if not present:
        return None
    vec = np.zeros(len(code_to_idx), dtype=np.float32)
    vec[present] = 1.0
    return vec


def load_signal(record_path: str) -> np.ndarray:
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal.astype(np.float32)
    if signal.shape[1] != NUM_LEADS:
        raise ValueError(f"Record {record_path} has {signal.shape[1]} leads, expected {NUM_LEADS}.")
    if np.isnan(signal).any():
        signal = np.nan_to_num(signal, nan=0.0)
    return signal


# ===========================================================================
# Filtering
# ===========================================================================

def bandpass_filter(signal: np.ndarray, lowcut: float = PNConfig.BANDPASS_LOW,
                     highcut: float = PNConfig.BANDPASS_HIGH, fs: int = FS,
                     order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    filtered = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        filtered[:, i] = filtfilt(b, a, signal[:, i])
    return np.nan_to_num(filtered, nan=0.0).astype(np.float32)


def notch_filter(signal: np.ndarray, freq: float = NOTCH_FREQ, fs: int = FS,
                  quality: float = NOTCH_Q) -> np.ndarray:
    b, a = iirnotch(freq, quality, fs)
    filtered = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        filtered[:, i] = filtfilt(b, a, signal[:, i])
    return np.nan_to_num(filtered, nan=0.0).astype(np.float32)


def wavelet_denoise(signal: np.ndarray, wavelet: str = PNConfig.WAVELET,
                     level: int = PNConfig.WAVELET_LEVEL,
                     threshold_mode: str = "soft") -> np.ndarray:
    denoised = np.zeros_like(signal)
    for lead_idx in range(signal.shape[1]):
        lead_signal = signal[:, lead_idx]
        coeffs = pywt.wavedec(lead_signal, wavelet=wavelet, level=level)
        detail_finest = coeffs[-1]
        sigma = np.median(np.abs(detail_finest)) / 0.6745
        if not np.isfinite(sigma) or sigma < 1e-12:
            sigma = 0.0
        threshold = sigma * np.sqrt(2 * np.log(len(lead_signal)))
        denoised_coeffs = [coeffs[0]] + [
            pywt.threshold(d, value=threshold, mode=threshold_mode) for d in coeffs[1:]
        ]
        reconstructed = pywt.waverec(denoised_coeffs, wavelet=wavelet)
        reconstructed = np.nan_to_num(reconstructed, nan=0.0, posinf=0.0, neginf=0.0)
        denoised[:, lead_idx] = reconstructed[: len(lead_signal)]
    return denoised.astype(np.float32)


# ===========================================================================
# Signal quality screening
# ===========================================================================

def detect_flatline_leads(signal: np.ndarray, std_thresh: float = FLATLINE_STD_THRESH) -> np.ndarray:
    """True per lead = flatline (near-zero variance -> likely lead-off)."""
    stds = np.std(signal, axis=0)
    return stds < std_thresh


def detect_step_artifacts(signal: np.ndarray, fs: int = FS,
                           window_ms: float = STEP_ARTIFACT_WINDOW_MS,
                           z_thresh: float = STEP_ARTIFACT_Z) -> np.ndarray:
    """
    True per lead = contains a step artifact (sudden, sustained baseline
    jump typical of a lead disconnecting/reconnecting mid-recording).

    Operates on the MEAN of consecutive non-overlapping windows (default
    2 s, i.e. several beats wide), not the raw sample-to-sample derivative
    — a single QRS complex is a transient spike that mostly averages away
    within a multi-beat window, whereas a real lead-off step shifts the
    window mean itself and persists across the window boundary. A window
    that's too short (comparable to one R-R interval) still picks up
    beat-count-per-window fluctuation, so this needs several beats per
    window to be stable.

    Thresholding is a robust z-score (median/MAD) of the window-to-window
    mean differences *within that lead*, not an absolute noise-based
    threshold — this adapts to each lead's own normal QRS-driven
    window-mean fluctuation and flags only genuine outliers, which is what
    an actual disconnect step looks like relative to a recording's own
    baseline variability.
    """
    win = max(1, int(round(window_ms / 1000.0 * fs)))
    n_windows = signal.shape[0] // win
    flagged = np.zeros(signal.shape[1], dtype=bool)
    if n_windows < 4:
        return flagged  # recording too short to judge sustained shifts

    trimmed = signal[: n_windows * win]
    windowed = trimmed.reshape(n_windows, win, signal.shape[1])
    window_means = windowed.mean(axis=1)  # (n_windows, num_leads)

    for i in range(signal.shape[1]):
        d = np.diff(window_means[:, i])
        med = np.median(d)
        mad = np.median(np.abs(d - med)) + 1e-8
        robust_z = 0.6745 * (d - med) / mad
        flagged[i] = np.any(np.abs(robust_z) > z_thresh)
    return flagged


def screen_signal_quality(signal: np.ndarray) -> Dict[str, object]:
    """
    Per-lead + recording-level quality gate.

    Returns
    -------
    dict with:
      lead_valid : bool array (NUM_LEADS,), True = usable
      recording_ok : bool, whether to proceed with this recording at all
      reason : str, rejection reason if recording_ok is False
    """
    flatline = detect_flatline_leads(signal)
    step = detect_step_artifacts(signal)
    lead_valid = ~(flatline | step)

    if not lead_valid[LEAD_II_IDX]:
        return {"lead_valid": lead_valid, "recording_ok": False,
                "reason": "lead II unusable (flatline/step artifact)"}
    if lead_valid.sum() < MIN_VALID_LEADS:
        return {"lead_valid": lead_valid, "recording_ok": False,
                "reason": f"only {int(lead_valid.sum())}/{NUM_LEADS} leads valid"}
    return {"lead_valid": lead_valid, "recording_ok": True, "reason": ""}


# ===========================================================================
# R-peak detection
# ===========================================================================

def _pan_tompkins_r_peaks(lead_signal: np.ndarray, fs: int = FS) -> np.ndarray:
    """Simplified Pan-Tompkins R-peak detector on a single lead."""
    nyq = 0.5 * fs
    b, a = butter(2, [5.0 / nyq, 15.0 / nyq], btype="band")
    bp = filtfilt(b, a, lead_signal)

    deriv = np.diff(bp, prepend=bp[0])
    squared = deriv ** 2

    win = max(1, int(round(0.150 * fs)))  # 150 ms moving window integration
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")

    distance = max(1, int(round(R_PEAK_REFRACTORY_MS / 1000.0 * fs)))
    height = np.mean(integrated) + 0.5 * np.std(integrated)
    peak_idx, _ = find_peaks(integrated, distance=distance, height=height)

    # Refine: snap each detected peak to the true local max of the raw
    # bandpassed signal within +/- 50 ms, so windows are R-peak-centred
    # rather than centred on the integrated-signal peak.
    refine_win = max(1, int(round(0.050 * fs)))
    refined = []
    for p in peak_idx:
        lo, hi = max(0, p - refine_win), min(len(lead_signal), p + refine_win)
        if hi > lo:
            refined.append(lo + int(np.argmax(np.abs(lead_signal[lo:hi]))))
    return np.array(sorted(set(refined)), dtype=int)


def _wavelet_r_peaks(lead_signal: np.ndarray, fs: int = FS) -> np.ndarray:
    """Fallback QRS detector: wavelet-enhanced peak picking (used when
    Pan-Tompkins yields too few beats, e.g. very noisy lead)."""
    coeffs = pywt.swt(lead_signal, wavelet="db4", level=2, trim_approx=True)
    qrs_enhanced = coeffs[0]  # coarsest detail-ish component, QRS-dominant band
    qrs_enhanced = np.abs(qrs_enhanced)
    distance = max(1, int(round(R_PEAK_REFRACTORY_MS / 1000.0 * fs)))
    height = np.mean(qrs_enhanced) + 1.0 * np.std(qrs_enhanced)
    peak_idx, _ = find_peaks(qrs_enhanced, distance=distance, height=height)
    return np.array(sorted(peak_idx), dtype=int)


def detect_r_peaks(lead_signal: np.ndarray, fs: int = FS,
                    min_beats: int = MIN_BEATS_PER_RECORD) -> Tuple[np.ndarray, str]:
    """Returns (r_peak_indices, method_used)."""
    peaks = _pan_tompkins_r_peaks(lead_signal, fs)
    if len(peaks) >= min_beats:
        return peaks, "pan_tompkins"
    peaks_fallback = _wavelet_r_peaks(lead_signal, fs)
    if len(peaks_fallback) > len(peaks):
        return peaks_fallback, "wavelet_fallback"
    return peaks, "pan_tompkins"


def cross_validate_r_peaks(peaks_ii: np.ndarray, peaks_v1: np.ndarray, fs: int = FS,
                            tol_ms: float = CROSS_VALIDATION_TOL_MS,
                            min_agreement: float = CROSS_VALIDATION_MIN_AGREEMENT
                            ) -> Tuple[np.ndarray, float]:
    """
    Keep only lead-II peaks that have a matching lead-V1 peak within
    tolerance (drops spurious detections). If agreement is too low (V1
    itself unreliable), fall back to the unfiltered lead-II peaks.

    Returns (validated_peaks, agreement_ratio).
    """
    if len(peaks_ii) == 0 or len(peaks_v1) == 0:
        return peaks_ii, 0.0
    tol = int(round(tol_ms / 1000.0 * fs))
    kept = []
    for p in peaks_ii:
        if np.any(np.abs(peaks_v1 - p) <= tol):
            kept.append(p)
    agreement = len(kept) / len(peaks_ii)
    if agreement < min_agreement:
        # V1 likely unreliable itself — don't let it veto lead II.
        return peaks_ii, agreement
    return np.array(kept, dtype=int), agreement


# ===========================================================================
# Beat windowing + normalization
# ===========================================================================

def extract_beats(signal: np.ndarray, r_peaks: np.ndarray,
                   pre_samples: int = PRE_R_SAMPLES,
                   post_samples: int = POST_R_SAMPLES) -> Tuple[np.ndarray, np.ndarray]:
    """
    Window each R-peak into (pre_samples + post_samples, num_leads).
    Boundary beats that would run off either end of the recording are
    dropped.

    Returns (beats, kept_r_peak_indices).
    """
    n = signal.shape[0]
    beats, kept = [], []
    for p in r_peaks:
        lo, hi = p - pre_samples, p + post_samples
        if lo < 0 or hi > n:
            continue
        beats.append(signal[lo:hi, :])
        kept.append(p)
    if not beats:
        return np.zeros((0, pre_samples + post_samples, signal.shape[1]), dtype=np.float32), np.array([], dtype=int)
    return np.stack(beats).astype(np.float32), np.array(kept, dtype=int)


def normalize_beats(beats: np.ndarray, ref_signal: np.ndarray) -> np.ndarray:
    """
    Per-lead z-score using mean/std computed once over the full filtered
    recording (ref_signal), applied to every beat window. Keeps normalization
    consistent across beats within a recording rather than per-window.
    """
    mean = np.mean(ref_signal, axis=0)              # (num_leads,)
    std = np.std(ref_signal, axis=0) + 1e-8          # (num_leads,)
    normalized = (beats - mean[None, None, :]) / std[None, None, :]
    return normalized.astype(np.float32)


# ===========================================================================
# Per-record processing
# ===========================================================================

def process_record(record_path: str, code_to_idx: Dict[str, int]) -> Optional[Dict[str, object]]:
    """
    Returns a dict with beats/label/metadata, or None if the record is
    skipped. `reason` is attached to any dict returned for a skip so the
    caller can tally rejection causes; on success `reason` is "".
    """
    record_id = os.path.basename(record_path)

    try:
        dx_codes = get_dx_codes(record_path)
    except Exception as exc:
        return {"skip": True, "reason": f"unreadable header: {exc}", "record_id": record_id}

    label_vec = assign_multi_label(dx_codes, code_to_idx)
    if label_vec is None:
        return {"skip": True, "reason": "no Dx code in whitelist", "record_id": record_id}

    try:
        signal = load_signal(record_path)
    except Exception as exc:
        return {"skip": True, "reason": f"unreadable signal: {exc}", "record_id": record_id}

    filtered = bandpass_filter(signal)
    filtered = notch_filter(filtered)
    if USE_WAVELET_DENOISE:
        filtered = wavelet_denoise(filtered)

    quality = screen_signal_quality(filtered)
    if not quality["recording_ok"]:
        return {"skip": True, "reason": f"quality: {quality['reason']}", "record_id": record_id}
    lead_valid = quality["lead_valid"]

    peaks_ii, method_ii = detect_r_peaks(filtered[:, LEAD_II_IDX])
    if len(peaks_ii) < MIN_BEATS_PER_RECORD:
        return {"skip": True, "reason": "insufficient R-peaks on lead II", "record_id": record_id}

    agreement = None
    if lead_valid[LEAD_V1_IDX]:
        peaks_v1, _ = detect_r_peaks(filtered[:, LEAD_V1_IDX])
        peaks_ii, agreement = cross_validate_r_peaks(peaks_ii, peaks_v1)
        if len(peaks_ii) < MIN_BEATS_PER_RECORD:
            return {"skip": True, "reason": "insufficient R-peaks after cross-validation", "record_id": record_id}

    beats, kept_peaks = extract_beats(filtered, peaks_ii)
    if beats.shape[0] < MIN_BEATS_PER_RECORD:
        return {"skip": True, "reason": "insufficient beats after windowing (boundary loss)", "record_id": record_id}

    beats = normalize_beats(beats, filtered)
    beats = np.nan_to_num(beats, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "skip": False,
        "reason": "",
        "record_id": record_id,
        "beats": beats,                      # (n_beats, window_len, num_leads)
        "label": label_vec,                  # (num_classes,) multi-hot
        "r_peaks": kept_peaks,                # (n_beats,) sample indices into filtered signal
        "lead_valid": lead_valid,             # (num_leads,) bool
        "r_peak_method": method_ii,
        "v1_agreement": agreement,
    }


# ===========================================================================
# Multi-label stratified split (simple combination-based fallback — swap
# for iterative-stratification / skmultilearn if you install it)
# ===========================================================================

def multilabel_stratified_split(
    record_ids: List[str], labels: np.ndarray,
    val_frac: float, test_frac: float, seed: int = PNConfig.RANDOM_SEED,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Stratifies by exact label-combination where that combination has enough
    members to split; singleton/rare combinations go straight to train (can't
    be meaningfully stratified, and dropping them would lose data for no
    reason).
    """
    rng = np.random.RandomState(seed)
    combo_key = ["_".join(map(str, row.astype(int))) for row in labels]
    combo_to_ids: Dict[str, List[int]] = {}
    for i, key in enumerate(combo_key):
        combo_to_ids.setdefault(key, []).append(i)

    train_idx, val_idx, test_idx = [], [], []
    for key, idxs in combo_to_ids.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        if n < 3:
            # Too few to split meaningfully — keep in train.
            train_idx.extend(idxs)
            continue
        n_val = max(1, int(round(n * val_frac)))
        n_test = max(1, int(round(n * test_frac)))
        n_val = min(n_val, n - 2)
        n_test = min(n_test, n - n_val - 1)
        val_idx.extend(idxs[:n_val])
        test_idx.extend(idxs[n_val:n_val + n_test])
        train_idx.extend(idxs[n_val + n_test:])

    to_ids = lambda idxs: [record_ids[i] for i in idxs]
    return to_ids(train_idx), to_ids(val_idx), to_ids(test_idx)


# ===========================================================================
# Pipeline entry point
# ===========================================================================

def run_preprocessing_pipeline(
    data_path: str = PNConfig.DATA_PATH,
    save_dir: str = PNConfig.PROCESSED_PATH,
    val_frac: float = PNConfig.VALIDATION_SPLIT,
    test_frac: float = PNConfig.TEST_SPLIT,
) -> None:
    PNConfig.create_output_dirs()
    beats_dir = os.path.join(save_dir, BEATS_SUBDIR)
    os.makedirs(beats_dir, exist_ok=True)

    logger.info("Discovering records...")
    record_paths = find_all_records(data_path)

    logger.info("Loading SNOMED-CT name table...")
    snomed_names = load_snomed_names()

    logger.info("Building label vocabulary (whitelist: SB/AFIB/SR/ST/SVT)...")
    code_to_idx, class_names, class_codes = build_label_vocab(record_paths, snomed_names)

    logger.info("Processing %d records...", len(record_paths))
    index_records: List[Dict[str, object]] = []
    skip_reasons: Counter = Counter()
    total_beats = 0

    for i, rp in enumerate(record_paths, 1):
        result = process_record(rp, code_to_idx)

        if result["skip"]:
            skip_reasons[result["reason"].split(":")[0]] += 1
        else:
            npz_path = os.path.join(beats_dir, f"{result['record_id']}.npz")
            np.savez_compressed(
                npz_path,
                beats=result["beats"],
                label=result["label"],
                r_peaks=result["r_peaks"],
                lead_valid=result["lead_valid"],
            )
            index_records.append({
                "record_id": result["record_id"],
                "npz_path": npz_path,
                "label": result["label"].tolist(),
                "n_beats": int(result["beats"].shape[0]),
                "r_peak_method": result["r_peak_method"],
                "v1_agreement": result["v1_agreement"],
            })
            total_beats += result["beats"].shape[0]

        if i % 500 == 0:
            logger.info("  ...%d/%d processed (%d kept, %d skipped)",
                        i, len(record_paths), len(index_records), sum(skip_reasons.values()))

    if not index_records:
        raise RuntimeError("No records survived preprocessing — check DATA_PATH and quality thresholds.")

    logger.info("Kept %d records (%d total beats), skipped %d.",
                len(index_records), total_beats, sum(skip_reasons.values()))
    logger.info("Skip reasons: %s", dict(skip_reasons))

    labels = np.array([r["label"] for r in index_records], dtype=np.float32)
    logger.info("Per-class positive count: %s",
                dict(zip(class_names, labels.sum(axis=0).astype(int).tolist())))

    record_ids = [r["record_id"] for r in index_records]
    id_to_record = {r["record_id"]: r for r in index_records}
    train_ids, val_ids, test_ids = multilabel_stratified_split(record_ids, labels, val_frac, test_frac)

    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_path = os.path.join(save_dir, f"index_{split_name}.json")
        with open(split_path, "w") as fh:
            json.dump([id_to_record[i] for i in ids], fh, indent=2)
        logger.info("Saved %-5s -> %s  (%d records)", split_name, split_path, len(ids))

    vocab_path = os.path.join(save_dir, "label_vocab.json")
    with open(vocab_path, "w") as fh:
        json.dump({"class_names": class_names, "class_codes": class_codes}, fh, indent=2)
    logger.info("Saved label vocabulary -> %s", vocab_path)

    PNConfig.load_label_vocab()
    logger.info("\n%s", PNConfig.summary())


if __name__ == "__main__":
    run_preprocessing_pipeline()