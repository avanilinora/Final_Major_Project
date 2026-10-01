from __future__ import annotations

"""
utils/preprocessing_physionet.py
=================================
End-to-end preprocessing pipeline for the PhysioNet 12-lead ECG-Arrhythmia
(Chapman-Shaoxing / Ningbo) dataset.

Unlike MIT-BIH, this dataset has NO beat-level R-peak annotations — each
recording carries one or more whole-recording SNOMED-CT diagnosis codes in
its WFDB header. So instead of segmenting beats around annotated R-peaks,
this pipeline treats each *recording* as one fixed-length sample.

Pipeline
--------
1. find_all_records()      — recursively locate every record under
                              <DATA_PATH>/WFDBRecords/
2. load_snomed_names()      — parse ConditionNames_SNOMED-CT.csv
3. get_dx_codes()           — read the "Dx:" comment line from a .hea header
4. build_label_vocab()      — scan all records once, keep the TOP_K most
                              frequent codes as classes
5. assign_label()           — map a record's Dx codes -> single class index
6. load_signal()            — wfdb.rdrecord -> (N, 12) float32
7. bandpass_filter() / wavelet_denoise() / normalize_signal()
   — identical design to the MIT-BIH track, generalised to N leads
8. fix_length()             — centre-crop / zero-pad to RECORD_LENGTH
9. run_preprocessing_pipeline() — orchestrates everything, saves
   X_train.npy / y_train.npy / ... and label_vocab.json

*** CHANGE: AF merged into AFIB (do not revert without re-checking) ***
AF (164890007) and AFIB (164889003) are clinically adjacent SNOMED codes,
and evaluation showed the trained model consistently confused them (AF
F1 ~0.50 despite having real support — a genuine class-separability
problem, unlike TWC/55827005/STTC which were dropped purely for having
too few samples). Rather than keep tuning the loss function against this,
`CODE_MERGE_MAP` below folds every AF-coded record into the AFIB class
BEFORE label-vocab counting and BEFORE label assignment, so the two
become one class end-to-end. This is applied in exactly two places:
`build_label_vocab` (so AF's occurrences count toward AFIB's frequency,
not their own) and `process_record` (so individual records get labelled
AFIB, not AF). AF no longer appears as its own class — do not also list
"164890007" in `exclude_codes`, it would be redundant (and harmless,
since after merging it never reaches the vocabulary as its own code
anyway — but it's clearer to leave it out).

Usage
-----
    from utils.preprocessing_physionet import run_preprocessing_pipeline
    run_preprocessing_pipeline()

    # or from the command line:
    python utils/preprocessing_physionet.py
"""


import csv
import glob
import json
import logging
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pywt
import wfdb
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split

from config_physionet import PhysioNetConfig as PNConfig

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Code merge map — records coded as the KEY are treated as the VALUE from
# here on. Applied once, consistently, at both counting and assignment time.
# ---------------------------------------------------------------------------
CODE_MERGE_MAP: Dict[str, str] = {
    "164890007": "164889003",  # AF -> AFIB
}


def _normalize_dx_codes(dx_codes: List[str]) -> List[str]:
    """Apply CODE_MERGE_MAP to a record's raw Dx codes."""
    return [CODE_MERGE_MAP.get(c, c) for c in dx_codes]


# ===========================================================================
# 1. Record discovery
# ===========================================================================

def find_all_records(data_path: str = PNConfig.DATA_PATH) -> List[str]:
    """
    Recursively find every WFDB record under ``<data_path>/WFDBRecords``.

    The dataset nests records several directories deep
    (``WFDBRecords/01/010/JS00001.hea``), so this globs recursively rather
    than assuming a fixed depth.

    Parameters
    ----------
    data_path : str
        Root directory containing the ``WFDBRecords`` folder.

    Returns
    -------
    list[str]
        Record base paths *without* file extension (e.g. ``.../JS00001``),
        suitable for ``wfdb.rdrecord`` / ``wfdb.rdheader``.
    """
    records_root = os.path.join(data_path, PNConfig.RECORDS_SUBDIR)
    if not os.path.isdir(records_root):
        raise FileNotFoundError(
            f"Could not find {records_root}. "
            "Check PhysioNetConfig.DATA_PATH points at the folder that "
            "contains WFDBRecords/."
        )

    hea_files = glob.glob(
        os.path.join(records_root, "**", "*.hea"), recursive=True
    )
    record_paths = sorted(p[:-4] for p in hea_files)  # strip ".hea"

    logger.info("Found %d records under %s", len(record_paths), records_root)
    return record_paths


# ===========================================================================
# 2. SNOMED-CT name lookup
# ===========================================================================

def load_snomed_names(
    csv_path: str = PNConfig.CONDITION_NAMES_CSV,
) -> Dict[str, str]:
    """
    Parse ``ConditionNames_SNOMED-CT.csv`` into a ``{code: display_name}`` map.

    Column names in the released CSV vary slightly between mirrors, so this
    looks for the first column containing "snomed" (the code) and the first
    remaining column containing "name" or "acronym" (the label). If neither
    is found it falls back to the first two columns.

    Parameters
    ----------
    csv_path : str
        Path to ``ConditionNames_SNOMED-CT.csv``.

    Returns
    -------
    dict[str, str]
        Maps SNOMED-CT code (as string) -> human-readable condition name.
    """
    if not os.path.exists(csv_path):
        logger.warning(
            "ConditionNames_SNOMED-CT.csv not found at %s — "
            "class names will fall back to raw SNOMED codes.",
            csv_path,
        )
        return {}

    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []

        code_col = next(
            (c for c in fieldnames if "snomed" in c.lower()), fieldnames[0]
        )
        name_col = next(
            (
                c
                for c in fieldnames
                if c != code_col and ("name" in c.lower() or "acronym" in c.lower())
            ),
            fieldnames[1] if len(fieldnames) > 1 else code_col,
        )

        mapping: Dict[str, str] = {}
        for row in reader:
            code = str(row[code_col]).strip()
            name = str(row[name_col]).strip()
            if code:
                mapping[code] = name

    logger.info(
        "Loaded %d SNOMED-CT code -> name mappings (code_col=%r, name_col=%r)",
        len(mapping), code_col, name_col,
    )
    return mapping


# ===========================================================================
# 3. Diagnosis code extraction
# ===========================================================================

def get_dx_codes(record_path: str) -> List[str]:
    """
    Extract the SNOMED-CT diagnosis codes listed in a record's header.

    The header's free-text comments include a line such as
    ``#Dx: 426177001,164934002`` — this function locates that line
    (case-insensitively, tolerant of a missing "#") and splits it.

    Parameters
    ----------
    record_path : str
        Record base path without extension.

    Returns
    -------
    list[str]
        Diagnosis codes as strings, in the order they appear in the header.
        Empty list if no Dx line is found.
    """
    header = wfdb.rdheader(record_path)
    comments = header.comments or []

    for line in comments:
        stripped = line.lstrip("#").strip()
        if stripped.lower().startswith("dx"):
            _, _, value = stripped.partition(":")
            codes = [c.strip() for c in value.split(",") if c.strip()]
            return codes

    return []


# ===========================================================================
# 4. Label vocabulary
# ===========================================================================

def build_label_vocab(
    record_paths: List[str],
    snomed_names: Dict[str, str],
    top_k: Optional[int] = PNConfig.TOP_K_CLASSES,
    min_count: int = PNConfig.MIN_CLASS_COUNT,
    exclude_codes: Optional[List[str]] = None,
    include_codes: Optional[List[str]] = None,
) -> Tuple[Dict[str, int], List[str], List[str]]:
    """
    Scan every record's Dx codes once and select the classes to train on.

    *** include_codes — use this, not exclude_codes, going forward ***
    `top_k`/`exclude_codes` rank classes by RAW Dx-mention count across all
    records (a code counts even when it's only a secondary/comorbid
    diagnosis). That count does not predict how many records will actually
    get that code as their PRIMARY assigned label (`assign_label` takes the
    first matching code in header order) — in practice several codes with
    high raw mention counts (seen twice now: TWC/STTC/55827005, then
    RBBB/STDD/ALS/VPB) turned out to have single-digit-to-low-double-digit
    real primary-label support after assignment, once the previous batch
    of weak codes was excluded and the next-highest-raw-frequency codes
    took their place in the top-k. Excluding codes one at a time is
    whack-a-mole against this ranking metric being the wrong one.

    If `include_codes` is given, it is used VERBATIM as the class list —
    `top_k`, `min_count`, and `exclude_codes` are all ignored. 

    Parameters
    ----------
    record_paths : list[str]
        All discovered record base paths.
    snomed_names : dict[str, str]
        Code -> display name, from ``load_snomed_names``.
    top_k : int, optional
        Keep only the ``top_k`` most frequent codes. ``None`` keeps every
        code that appears at least ``min_count`` times. Ignored if
        `include_codes` is given.
    min_count : int
        Minimum occurrence count for a code to be considered a class
        (ignored when ``top_k`` is set and fewer than ``top_k`` codes clear
        this bar — in that case the top_k most frequent codes are kept
        regardless).
    exclude_codes : list[str], optional
        SNOMED-CT codes to exclude from consideration entirely, even if
        they would otherwise rank in the top_k. Use this for codes that
        are frequent as SECONDARY mentions across many records but rarely
        win the primary single-label assignment (``assign_label`` takes
        the first matching code in header order) — top_k ranks by raw
        mention count, which is the wrong metric for those codes, so
        excluding them explicitly is more reliable than adjusting top_k.

    Returns
    -------
    code_to_idx : dict[str, int]
        SNOMED-CT code -> integer class index.
    class_names : list[str]
        Display name per class index, e.g. ``["Sinus Rhythm", "AFib", ...]``.
    class_codes : list[str]
        Raw SNOMED-CT code per class index, same order as ``class_names``.
    """
    exclude_set = set(exclude_codes or [])
    counter: Counter = Counter()
    skipped_no_dx = 0

    for rp in record_paths:
        try:
            codes = get_dx_codes(rp)
        except Exception as exc:  # corrupt/missing header
            logger.warning("Could not read header for %s: %s", rp, exc)
            continue

        if not codes:
            skipped_no_dx += 1
            continue

        # Merge codes (e.g. AF -> AFIB) BEFORE counting, so merged codes
        # contribute to their canonical code's frequency, not their own.
        codes = _normalize_dx_codes(codes)
        codes = [c for c in codes if c not in exclude_set]
        if codes:
            counter.update(codes)

    if skipped_no_dx:
        logger.info("%d records had no Dx codes in their header.", skipped_no_dx)

    if include_codes is not None:
        # Explicit whitelist — use exactly these codes, in the given order,
        # regardless of top_k/min_count. Still pulls the real count from
        # `counter` so the logged (n=...) is accurate.
        most_common = [(code, counter.get(code, 0)) for code in include_codes]
    elif top_k is not None:
        most_common = counter.most_common(top_k)
    else:
        most_common = [
            (code, n) for code, n in counter.most_common() if n >= min_count
        ]

    code_to_idx: Dict[str, int] = {}
    class_names: List[str] = []
    class_codes: List[str] = []

    for idx, (code, count) in enumerate(most_common):
        code_to_idx[code] = idx
        class_codes.append(code)
        class_names.append(snomed_names.get(code, code))
        logger.info("  class %2d : %-12s %-35s (n=%d)", idx, code,
                    snomed_names.get(code, "?"), count)

    logger.info("Kept %d classes out of %d distinct Dx codes seen.",
                len(class_names), len(counter))

    return code_to_idx, class_names, class_codes


def assign_label(
    dx_codes: List[str],
    code_to_idx: Dict[str, int],
) -> Optional[int]:
    """
    Map a record's diagnosis codes to a single primary class index.

    The first code (in header order, after CODE_MERGE_MAP normalisation —
    see module docstring) that is present in ``code_to_idx`` is used as
    the primary label. Records whose codes are *all* outside the kept
    vocabulary return ``None`` and should be dropped.

    Parameters
    ----------
    dx_codes : list[str]
        Codes extracted via ``get_dx_codes`` (NOT yet normalised — this
        function normalises internally so every caller gets the merge
        applied consistently).
    code_to_idx : dict[str, int]
        Vocabulary built by ``build_label_vocab``.

    Returns
    -------
    int or None
        Class index, or ``None`` if no code matched the kept vocabulary.

    Notes
    -----
    This dataset is genuinely multi-label (a recording can show, e.g., both
    sinus bradycardia AND a premature complex). Picking a single "primary"
    label is a simplification — if  multi-label modelling is needed, use
    ``assign_multi_label`` instead and switch the training loss to
    ``BCEWithLogitsLoss``.
    """
    for code in _normalize_dx_codes(dx_codes):
        if code in code_to_idx:
            return code_to_idx[code]
    return None


def assign_multi_label(
    dx_codes: List[str],
    code_to_idx: Dict[str, int],
) -> Optional[np.ndarray]:
    """
    Map a record's diagnosis codes to a multi-hot label vector.

    Returns
    -------
    np.ndarray, shape (num_classes,), dtype float32
        Multi-hot vector, or ``None`` if no code matched the vocabulary.
    """
    dx_codes = _normalize_dx_codes(dx_codes)
    present = [code_to_idx[c] for c in dx_codes if c in code_to_idx]
    if not present:
        return None
    vec = np.zeros(len(code_to_idx), dtype=np.float32)
    vec[present] = 1.0
    return vec


# ===========================================================================
# 5. Signal loading
# ===========================================================================

def load_signal(record_path: str) -> np.ndarray:
    """
    Load the raw 12-lead signal for one record.

    Parameters
    ----------
    record_path : str
        Record base path without extension.

    Returns
    -------
    np.ndarray, shape (N, 12)
        Raw signal, float32, leads in header order.

    Raises
    ------
    ValueError
        If the record does not contain exactly ``PNConfig.NUM_LEADS`` leads.
    """
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal.astype(np.float32)  # (N, num_leads)

    if signal.shape[1] != PNConfig.NUM_LEADS:
        raise ValueError(
            f"Record {record_path} has {signal.shape[1]} leads, "
            f"expected {PNConfig.NUM_LEADS}."
        )

    # Replace any NaNs (occasionally present from lead disconnects) with 0
    # before filtering, so a single bad sample doesn't propagate via filtfilt.
    if np.isnan(signal).any():
        signal = np.nan_to_num(signal, nan=0.0)

    return signal


# ===========================================================================
# 6. Filtering / denoising / normalisation
#    (same design as the MIT-BIH track, generalised to N leads — these were
#    already lead-count-agnostic, just re-pointed at PhysioNetConfig)
# ===========================================================================

def bandpass_filter(
    signal: np.ndarray,
    lowcut: float = PNConfig.BANDPASS_LOW,
    highcut: float = PNConfig.BANDPASS_HIGH,
    fs: int = PNConfig.SAMPLING_FREQ,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter, applied per lead."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")

    filtered = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        filtered[:, i] = filtfilt(b, a, signal[:, i])
    filtered = np.nan_to_num(filtered, nan=0.0)

    return filtered.astype(np.float32)


def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = PNConfig.WAVELET,
    level: int = PNConfig.WAVELET_LEVEL,
    threshold_mode: str = "soft",
) -> np.ndarray:
    """Wavelet-denoise each lead independently (DWT soft-thresholding)."""
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
            pywt.threshold(d, value=threshold, mode=threshold_mode)
            for d in coeffs[1:]
        ]
        reconstructed = pywt.waverec(denoised_coeffs, wavelet=wavelet)
        reconstructed = np.nan_to_num(reconstructed, nan=0.0, posinf=0.0,neginf=0.0,)

        denoised[:, lead_idx] = reconstructed[:len(lead_signal)]

    return denoised.astype(np.float32)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Z-score normalisation, per lead."""
    normalized = np.zeros_like(signal)

    for lead_idx in range(signal.shape[1]):
        mean = np.mean(signal[:, lead_idx])
        std = np.std(signal[:, lead_idx]) + 1e-8
        normalized[:, lead_idx] = (signal[:, lead_idx] - mean) / std

    return normalized.astype(np.float32)


# ===========================================================================
# 7. Fixed-length windowing
# ===========================================================================

def fix_length(
    signal: np.ndarray,
    target_len: int = PNConfig.RECORD_LENGTH,
) -> np.ndarray:
    """
    Centre-crop or zero-pad a recording to a fixed number of samples.

    Recordings in this dataset vary in duration, but the CNN/BiLSTM
    architectures need a fixed input length. Centre-cropping (rather than
    taking the first N samples) keeps the most "typical" portion of longer
    recordings and avoids systematically discarding trailing beats.

    Parameters
    ----------
    signal : np.ndarray, shape (N, num_leads)
        Preprocessed signal.
    target_len : int
        Desired output length in samples.

    Returns
    -------
    np.ndarray, shape (target_len, num_leads)
    """
    n = signal.shape[0]

    if n == target_len:
        return signal

    if n > target_len:
        start = (n - target_len) // 2
        return signal[start:start + target_len]

    # n < target_len: zero-pad symmetrically
    pad_total = target_len - n
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    return np.pad(signal, ((pad_before, pad_after), (0, 0)), mode="constant")


# ===========================================================================
# 8. Per-record processing
# ===========================================================================

def process_record(
    record_path: str,
    code_to_idx: Dict[str, int],
) -> Optional[Tuple[np.ndarray, int]]:
    """
    Run the full preprocessing pipeline on one record.

    Returns
    -------
    (X, y) or None
        ``X`` has shape ``(RECORD_LENGTH, NUM_LEADS)``, ``y`` is an int
        class index. Returns ``None`` if the record should be skipped
        (unreadable, wrong lead count, or no Dx code in the kept vocabulary).
    """
    try:
        dx_codes = get_dx_codes(record_path)
        label = assign_label(dx_codes, code_to_idx)  # normalises internally
        if label is None:
            return None

        signal = load_signal(record_path)
        signal = bandpass_filter(signal)
        signal = wavelet_denoise(signal)
        signal = normalize_signal(signal)
        signal = fix_length(signal)

        # Remove any remaining NaN/Inf values
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        return signal.astype(np.float32), label

    except Exception as exc:
        logger.warning("Skipping %s: %s", record_path, exc)
        return None


# ===========================================================================
# 9. High-level pipeline entry-point
# ===========================================================================

def run_preprocessing_pipeline(
    data_path: str = PNConfig.DATA_PATH,
    save_dir: str = PNConfig.PROCESSED_PATH,
) -> None:
    """
    Discover records, build the label vocabulary, process every record, and
    save train/val/test ``.npy`` arrays plus ``label_vocab.json``.

    Splitting is stratified by class label at the RECORD level (there is
    one recording per patient in this dataset, so this is also a
    patient-level split — unlike inter-beat MIT-BIH splitting, there is no
    risk of beats from the same patient leaking across splits).
    """
    PNConfig.create_output_dirs()

    logger.info("Discovering records...")
    record_paths = find_all_records(data_path)

    logger.info("Loading SNOMED-CT name table...")
    snomed_names = load_snomed_names()

    logger.info("Building label vocabulary (explicit whitelist)...")
    # Switched from exclude_codes to an explicit whitelist after this
    # turned into whack-a-mole: excluding TWC/55827005/STTC (weak on raw
    # count) let RBBB/STDD/ALS/VPB rise into the top-9 by raw mention
    # count, and THEY turned out to have single-digit real primary-label
    # support too (see build_label_vocab's docstring). These 5 codes are
    # the ones confirmed, from an actual preprocessing run's final "Class
    # distribution" log, to have real support (>500 primary-assigned
    # samples each): SB, AFIB (post AF-merge), SR, ST, SVT.
    #
    # AF (164890007) is not listed separately — it's merged into AFIB via
    # CODE_MERGE_MAP (see module docstring), confirmed working: AFIB rose
    # from ~1780 to ~2313 after the merge in the last real run.
    code_to_idx, class_names, class_codes = build_label_vocab(
        record_paths, snomed_names,
        include_codes=[
            "426177001",  # SB
            "164889003",  # AFIB
            "426783006",  # SR
            "427084000",  # ST
            "426761007",  # SVT
        ],
    )

    logger.info("Processing %d records (this can take a while)...", len(record_paths))
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    n_skipped = 0

    for i, rp in enumerate(record_paths, 1):
        result = process_record(rp, code_to_idx)
        if result is None:
            n_skipped += 1
        else:
            X, y = result
            if np.isnan(X).any():
             logger.warning("NaN detected in %s", rp)
            X_list.append(X)
            y_list.append(y)

        if i % 500 == 0:
            logger.info("  ...%d / %d records processed (%d kept, %d skipped)",
                        i, len(record_paths), len(X_list), n_skipped)

    if not X_list:
        raise RuntimeError(
            "No records survived preprocessing — check DATA_PATH, "
            "TOP_K_CLASSES, and that ConditionNames_SNOMED-CT.csv matches "
            "the codes found in your headers."
        )

    X = np.stack(X_list)          # (N, RECORD_LENGTH, NUM_LEADS)
    y = np.array(y_list, dtype=np.int64)

    logger.info("Final dataset: %d records kept, %d skipped.", len(X), n_skipped)
    logger.info("Class distribution: %s",
                dict(zip(class_names, np.bincount(y, minlength=len(class_names)).tolist())))

    # ------------------------------------------------------------------
    # Stratified train / val / test split (Memory-Efficient)
    # ------------------------------------------------------------------
    # Split index numbers instead of copying the massive 10GB array
    indices = np.arange(len(y))
    idx_train, idx_tmp, y_train, y_tmp = train_test_split(
        indices, y, 
        test_size=PNConfig.VALIDATION_SPLIT + PNConfig.TEST_SPLIT, 
        stratify=y, 
        random_state=PNConfig.RANDOM_SEED
    )

    relative_test_size = PNConfig.TEST_SPLIT / (
        PNConfig.VALIDATION_SPLIT + PNConfig.TEST_SPLIT
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_tmp, y_tmp,
        test_size=relative_test_size,
        stratify=y_tmp,
        random_state=PNConfig.RANDOM_SEED,
    )

    # ------------------------------------------------------------------
    # Save directly to disk from index slices to save RAM
    # ------------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)

    # Slice and save X_train
    path = os.path.join(save_dir, "X_train.npy")
    np.save(path, X[idx_train])
    logger.info("Saved X_train    -> %s  shape=%s", path, (len(idx_train), *X.shape[1:]))

    # Slice and save X_val
    path = os.path.join(save_dir, "X_val.npy")
    np.save(path, X[idx_val])
    logger.info("Saved X_val      -> %s  shape=%s", path, (len(idx_val), *X.shape[1:]))

    # Slice and save X_test
    path = os.path.join(save_dir, "X_test.npy")
    np.save(path, X[idx_test])
    logger.info("Saved X_test     -> %s  shape=%s", path, (len(idx_test), *X.shape[1:]))

    # Save target arrays
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "y_val.npy"), y_val)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)

    # Free the large array from RAM
    del X

    vocab_path = os.path.join(save_dir, "label_vocab.json")
    with open(vocab_path, "w") as fh:
        json.dump(
            {"class_names": class_names, "class_codes": class_codes},
            fh, indent=2,
        )
    logger.info("Saved label vocabulary -> %s", vocab_path)

    PNConfig.load_label_vocab()
    logger.info("\n%s", PNConfig.summary())


if __name__ == "__main__":
    run_preprocessing_pipeline()
