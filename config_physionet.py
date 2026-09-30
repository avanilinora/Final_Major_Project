"""
config_physionet.py
====================
Configuration for the PhysioNet 12-lead ECG-Arrhythmia (Chapman-Shaoxing /
Ningbo) record-level rhythm/diagnosis classification track.

This is intentionally a SEPARATE class from ``config.Config`` — the MIT-BIH
track and this track differ in sampling rate, lead count, and label
granularity (beat vs. whole recording). Keeping them independent means
running one pipeline can never silently change global state the other
pipeline depends on.

Usage
-----
    from config_physionet import PhysioNetConfig as PNConfig
    device = PNConfig.DEVICE
"""
from __future__ import annotations

import json
import os

import torch


class PhysioNetConfig:
    """Project-wide configuration for the PhysioNet 12-lead track."""

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    RANDOM_SEED: int = 42

    # ------------------------------------------------------------------
    # Signal properties
    # ------------------------------------------------------------------
    NUM_LEADS: int = 12
    LEAD_NAMES: list = [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6",
    ]
    SAMPLING_FREQ: int = 500                    # Hz — native rate of this dataset
    RECORD_SECONDS: float = 10.0                # fixed window after crop/pad
    RECORD_LENGTH: int = int(SAMPLING_FREQ * RECORD_SECONDS)  # 5000 samples

    # ------------------------------------------------------------------
    # Preprocessing (same filter design as the MIT-BIH track)
    # ------------------------------------------------------------------
    BANDPASS_LOW: float = 0.5
    BANDPASS_HIGH: float = 45.0
    WAVELET: str = "db4"
    WAVELET_LEVEL: int = 4

    # ------------------------------------------------------------------
    # Label taxonomy
    # ------------------------------------------------------------------
    # Number of most-frequent SNOMED-CT codes to keep as classes. Records
    # whose diagnoses are ALL outside this set are dropped from the dataset.
    # Set to None to keep every code that appears at least MIN_CLASS_COUNT
    # times (produces a much longer, more imbalanced label list).
    TOP_K_CLASSES: int = 9
    MIN_CLASS_COUNT: int = 50
    MULTI_LABEL: bool = False   # False = single primary diagnosis per recording

    # Populated by utils/preprocessing_physionet.py::build_label_vocab, then
    # persisted to <PROCESSED_PATH>/label_vocab.json and reloaded here via
    # `PhysioNetConfig.load_label_vocab()` before training/evaluation.
    NUM_CLASSES: int = 9
    CLASS_NAMES: list = []
    CLASS_CODES: list = []   # raw SNOMED-CT codes, same order as CLASS_NAMES

    # ------------------------------------------------------------------
    # Training hyper-parameters
    # ------------------------------------------------------------------
    BATCH_SIZE: int = 64        # smaller than MIT-BIH: 12 leads x 5000 samples/record
    LEARNING_RATE: float = 5e-4
    WEIGHT_DECAY: float = 1e-4
    NUM_EPOCHS: int = 100
    EARLY_STOPPING_PATIENCE: int = 20
    VALIDATION_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    CNN_FILTERS: list = [32, 64, 128]
    CNN_KERNEL_SIZES: list = [7, 5, 3]   # slightly wider: signal is 500Hz, not 360Hz
    LSTM_HIDDEN_SIZE: int = 128
    LSTM_NUM_LAYERS: int = 1
    ATTENTION_HIDDEN_SIZE: int = 64
    LEAD_ATTENTION_HIDDEN_SIZE: int = 32
    FC_DROPOUT: float = 0.3

    # ------------------------------------------------------------------
    # File-system paths
    # ------------------------------------------------------------------
    _BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

    # Point this at the folder that CONTAINS "WFDBRecords/" and
    # "ConditionNames_SNOMED-CT.csv" (i.e. wherever you extracted the
    # physionet.org download).
    DATA_PATH: str = os.path.join(_BASE_DIR, "data", "Physionet_dataset2")
    RECORDS_SUBDIR: str = "WFDBRecords"
    CONDITION_NAMES_CSV: str = os.path.join(
        DATA_PATH, "ConditionNames_SNOMED-CT.csv"
    )

    PROCESSED_PATH: str = os.path.join(_BASE_DIR, "data", "processed_physionet2")
    CHECKPOINT_PATH: str = os.path.join(
        _BASE_DIR, "outputs", "checkpoints_physionet2"
    )
    CONFUSION_MATRIX_PATH: str = os.path.join(
        _BASE_DIR, "outputs", "confusion_matrices_physionet2"
    )
    PLOTS_PATH: str = os.path.join(_BASE_DIR, "outputs", "plots_physionet2")

    # ------------------------------------------------------------------
    # Compute device
    # ------------------------------------------------------------------
    DEVICE: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @classmethod
    def create_output_dirs(cls) -> None:
        """Create all required output directories if they don't yet exist."""
        for path in (
            cls.DATA_PATH,
            cls.PROCESSED_PATH,
            cls.CHECKPOINT_PATH,
            cls.CONFUSION_MATRIX_PATH,
            cls.PLOTS_PATH,
        ):
            os.makedirs(path, exist_ok=True)

    @classmethod
    def load_label_vocab(cls) -> None:
        """
        Load ``CLASS_NAMES`` / ``CLASS_CODES`` / ``NUM_CLASSES`` from the
        ``label_vocab.json`` file written by the preprocessing pipeline.

        Call this once, right after ``create_output_dirs()``, in every
        training/evaluation/inference script for this track — the vocab
        is only known after preprocessing has actually scanned the data.
        """
        vocab_path = os.path.join(cls.PROCESSED_PATH, "label_vocab.json")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"label_vocab.json not found at {vocab_path}.\n"
                "Run `python utils/preprocessing_physionet.py` first."
            )
        with open(vocab_path) as fh:
            vocab = json.load(fh)
        cls.CLASS_NAMES = vocab["class_names"]
        cls.CLASS_CODES = vocab["class_codes"]
        cls.NUM_CLASSES = len(cls.CLASS_NAMES)

    @classmethod
    def summary(cls) -> str:
        """Human-readable configuration summary."""
        lines = [
            "=" * 55,
            "  PhysioNet ECG-Arrhythmia Track — Configuration",
            "=" * 55,
            f"  Device            : {cls.DEVICE}",
            f"  Random seed       : {cls.RANDOM_SEED}",
            f"  Num leads         : {cls.NUM_LEADS}",
            f"  Sampling freq     : {cls.SAMPLING_FREQ} Hz",
            f"  Record length     : {cls.RECORD_LENGTH} samples "
            f"({cls.RECORD_SECONDS}s)",
            f"  Num classes       : {cls.NUM_CLASSES}  {cls.CLASS_NAMES}",
            f"  Batch size        : {cls.BATCH_SIZE}",
            f"  Learning rate     : {cls.LEARNING_RATE}",
            f"  Data path         : {cls.DATA_PATH}",
            f"  Processed path    : {cls.PROCESSED_PATH}",
            f"  Checkpoint path   : {cls.CHECKPOINT_PATH}",
            "=" * 55,
        ]
        return "\n".join(lines)
"""
config_physionet.py
===================
Single configuration store for the TrustECG PhysioNet 12-lead ECG-Arrhythmia
record-level classification pipeline.

This file combines the useful configuration from the old MIT-BIH `config.py`
and the PhysioNet-specific `config_physionet.py`. Only the PhysioNet settings
are active in this pipeline; the old MIT-BIH values are retained only where
they are useful as shared preprocessing/model defaults.

Usage
-----
    from config_physionet import PhysioNetConfig as Config
    device = Config.DEVICE
"""



import json
import os

import torch


class PhysioNetConfig:
    """Central configuration for the PhysioNet 12-lead ECG pipeline."""

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    RANDOM_SEED: int = 42

    # ------------------------------------------------------------------
    # Signal / dataset properties
    # ------------------------------------------------------------------
    NUM_LEADS: int = 12
    LEAD_NAMES: list[str] = [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6",
    ]

    SAMPLING_FREQ: int = 500
    RECORD_SECONDS: float = 10.0
    RECORD_LENGTH: int = int(SAMPLING_FREQ * RECORD_SECONDS)

    # ------------------------------------------------------------------
    # Label taxonomy
    # ------------------------------------------------------------------
    TOP_K_CLASSES: int = 9
    MIN_CLASS_COUNT: int = 50
    MULTI_LABEL: bool = False

    # These are populated from label_vocab.json after preprocessing.
    NUM_CLASSES: int = 9
    CLASS_NAMES: list[str] = []
    CLASS_CODES: list[str] = []

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    BANDPASS_LOW: float = 0.5
    BANDPASS_HIGH: float = 45.0
    WAVELET: str = "db4"
    WAVELET_LEVEL: int = 4

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 5e-4
    WEIGHT_DECAY: float = 1e-4
    NUM_EPOCHS: int = 100
    EARLY_STOPPING_PATIENCE: int = 20

    VALIDATION_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    CNN_FILTERS: list[int] = [32, 64, 128]
    CNN_KERNEL_SIZES: list[int] = [7, 5, 3]
    LSTM_HIDDEN_SIZE: int = 128
    LSTM_NUM_LAYERS: int = 1
    ATTENTION_HIDDEN_SIZE: int = 64
    LEAD_ATTENTION_HIDDEN_SIZE: int = 32
    FC_DROPOUT: float = 0.3

    # ------------------------------------------------------------------
    # File-system paths
    # ------------------------------------------------------------------
    _BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

    DATA_PATH: str = os.path.join(
        _BASE_DIR, "data", "Physionet_dataset2"
    )
    RECORDS_SUBDIR: str = "WFDBRecords"
    CONDITION_NAMES_CSV: str = os.path.join(
        DATA_PATH, "ConditionNames_SNOMED-CT.csv"
    )

    PROCESSED_PATH: str = os.path.join(
        _BASE_DIR, "data", "processed_physionet2"
    )
    CHECKPOINT_PATH: str = os.path.join(
        _BASE_DIR, "outputs", "checkpoints_physionet2"
    )
    CONFUSION_MATRIX_PATH: str = os.path.join(
        _BASE_DIR, "outputs", "confusion_matrices_physionet2"
    )
    PLOTS_PATH: str = os.path.join(
        _BASE_DIR, "outputs", "plots_physionet2"
    )

    # ------------------------------------------------------------------
    # Compute device
    # ------------------------------------------------------------------
    DEVICE: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @classmethod
    def create_output_dirs(cls) -> None:
        """Create directories required by the PhysioNet pipeline."""
        for path in (
            cls.DATA_PATH,
            cls.PROCESSED_PATH,
            cls.CHECKPOINT_PATH,
            cls.CONFUSION_MATRIX_PATH,
            cls.PLOTS_PATH,
        ):
            os.makedirs(path, exist_ok=True)

    @classmethod
    def load_label_vocab(cls) -> None:
        """
        Load the actual PhysioNet label vocabulary created during preprocessing.

        The preprocessing script must be run before training/evaluation.
        """
        vocab_path = os.path.join(
            cls.PROCESSED_PATH, "label_vocab.json"
        )

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"label_vocab.json not found at {vocab_path}.\n"
                "Run `python utils/preprocessing_physionet.py` first."
            )

        with open(vocab_path, encoding="utf-8") as fh:
            vocab = json.load(fh)

        cls.CLASS_NAMES = vocab["class_names"]
        cls.CLASS_CODES = vocab["class_codes"]
        cls.NUM_CLASSES = len(cls.CLASS_NAMES)

    @classmethod
    def summary(cls) -> str:
        """Return a human-readable configuration summary."""
        lines = [
            "=" * 60,
            "  TrustECG — PhysioNet 12-lead ECG-Arrhythmia",
            "=" * 60,
            f"  Device            : {cls.DEVICE}",
            f"  Random seed       : {cls.RANDOM_SEED}",
            f"  Num leads         : {cls.NUM_LEADS}",
            f"  Sampling freq     : {cls.SAMPLING_FREQ} Hz",
            f"  Record length     : {cls.RECORD_LENGTH} samples "
            f"({cls.RECORD_SECONDS}s)",
            f"  Num classes       : {cls.NUM_CLASSES}  {cls.CLASS_NAMES}",
            f"  Batch size        : {cls.BATCH_SIZE}",
            f"  Learning rate     : {cls.LEARNING_RATE}",
            f"  Weight decay      : {cls.WEIGHT_DECAY}",
            f"  Data path         : {cls.DATA_PATH}",
            f"  Processed path    : {cls.PROCESSED_PATH}",
            f"  Checkpoint path   : {cls.CHECKPOINT_PATH}",
            f"  Plots path        : {cls.PLOTS_PATH}",
            "=" * 60,
        ]
        return "\n".join(lines)


# Short alias so the rest of the PhysioNet pipeline can simply use `Config`.
Config = PhysioNetConfig
