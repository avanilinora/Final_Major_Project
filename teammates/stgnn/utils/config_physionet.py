"""
config_physionet.py — TrustECG STGNN track
===========================================
Config for the PhysioNet Chapman-Shaoxing / Ningbo 12-lead pipeline
(preprocessing_physionet_stgnn.py). Separate from config.py, which is the
MIT-BIH / AAMI single-lead track — the two datasets have different sampling
rates, lead counts, and label schemes, so they don't share a config class.
"""

import json
import os

import torch


class PhysioNetConfig:

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
    SAMPLING_FREQ: int = 500  # Hz

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    BANDPASS_LOW: float = 0.5    # Hz
    BANDPASS_HIGH: float = 40.0  # Hz
    NOTCH_FREQ: float = 50.0     # Hz — mains frequency; set to 60.0 if the
                                  # recording site uses 60 Hz mains.
    NOTCH_Q: float = 30.0
    WAVELET: str = "db4"
    WAVELET_LEVEL: int = 4
    USE_WAVELET_DENOISE: bool = True

    # ------------------------------------------------------------------
    # Signal quality screening
    # ------------------------------------------------------------------
    FLATLINE_STD_THRESH: float = 1e-3   # mV; lead std below this = flatline
    STEP_ARTIFACT_Z: float = 5.0        # robust z-score on windowed-mean diffs
    STEP_ARTIFACT_WINDOW_MS: float = 1000.0
    MIN_VALID_LEADS: int = 10           # of 12; below this, drop the recording

    # ------------------------------------------------------------------
    # R-peak detection / beat windowing
    # ------------------------------------------------------------------
    LEAD_II_IDX: int = 1   # index into LEAD_NAMES — primary R-peak lead
    LEAD_V1_IDX: int = 6   # cross-validation lead
    R_PEAK_REFRACTORY_MS: float = 250.0
    CROSS_VALIDATION_TOL_MS: float = 100.0
    CROSS_VALIDATION_MIN_AGREEMENT: float = 0.5  # below this, trust lead II alone
    MIN_BEATS_PER_RECORD: int = 3

    PRE_R_MS: float = 300.0
    POST_R_MS: float = 500.0

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    # Populated by load_label_vocab() after preprocessing writes
    # label_vocab.json; placeholders here reflect the intended whitelist
    # (SB, AFIB [AF merged in], SR, ST, SVT) so the config is usable before
    # that file exists.
    NUM_CLASSES: int = 5
    CLASS_NAMES: list = ["SB", "AFIB", "SR", "ST", "SVT"]
    CLASS_CODES: list = ["426177001", "164889003", "426783006", "427084000", "426761007"]

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    VALIDATION_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15

    # ------------------------------------------------------------------
    # File-system paths
    # ------------------------------------------------------------------
    _BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATA_PATH: str = r"C:\Users\Alma\OneDrive\Desktop\MAJOR_PRJ\data\raw"
    RECORDS_SUBDIR: str = "WFDBRecords"
    CONDITION_NAMES_CSV: str = os.path.join(DATA_PATH, "ConditionNames_SNOMED-CT.csv")

    PROCESSED_PATH: str = os.path.join(_BASE_DIR, "data", "processed_stgnn")
    CHECKPOINT_PATH: str = os.path.join(_BASE_DIR, "outputs", "checkpoints_stgnn")
    CONFUSION_MATRIX_PATH: str = os.path.join(_BASE_DIR, "outputs", "confusion_matrices_stgnn")
    PLOTS_PATH: str = os.path.join(_BASE_DIR, "outputs", "plots_stgnn")

    # ------------------------------------------------------------------
    # Compute device
    # ------------------------------------------------------------------
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @classmethod
    def create_output_dirs(cls) -> None:
        for path in [cls.DATA_PATH, cls.PROCESSED_PATH, cls.CHECKPOINT_PATH,
                     cls.CONFUSION_MATRIX_PATH, cls.PLOTS_PATH]:
            os.makedirs(path, exist_ok=True)

    @classmethod
    def load_label_vocab(cls) -> None:
        """
        Load class_names/class_codes from <PROCESSED_PATH>/label_vocab.json
        if it exists (written by run_preprocessing_pipeline), and sync
        NUM_CLASSES/CLASS_NAMES/CLASS_CODES to match. No-op with a warning
        if the file isn't there yet — the class-level defaults above still
        hold in that case.
        """
        vocab_path = os.path.join(cls.PROCESSED_PATH, "label_vocab.json")
        if not os.path.exists(vocab_path):
            print(f"[PhysioNetConfig] No label_vocab.json at {vocab_path} yet "
                  f"— using default CLASS_NAMES {cls.CLASS_NAMES}.")
            return
        with open(vocab_path) as fh:
            vocab = json.load(fh)
        cls.CLASS_NAMES = vocab["class_names"]
        cls.CLASS_CODES = vocab["class_codes"]
        cls.NUM_CLASSES = len(cls.CLASS_NAMES)

    @classmethod
    def summary(cls) -> str:
        lines = [
            "=" * 55,
            "  TrustECG — PhysioNet STGNN Configuration",
            "=" * 55,
            f"  Device              : {cls.DEVICE}",
            f"  Random seed         : {cls.RANDOM_SEED}",
            f"  Num leads           : {cls.NUM_LEADS}  {cls.LEAD_NAMES}",
            f"  Sampling freq       : {cls.SAMPLING_FREQ} Hz",
            f"  Bandpass            : {cls.BANDPASS_LOW}-{cls.BANDPASS_HIGH} Hz",
            f"  Notch               : {cls.NOTCH_FREQ} Hz (Q={cls.NOTCH_Q})",
            f"  Wavelet             : {cls.WAVELET} (level {cls.WAVELET_LEVEL}, "
            f"enabled={cls.USE_WAVELET_DENOISE})",
            f"  Flatline std thresh : {cls.FLATLINE_STD_THRESH} mV",
            f"  Step artifact       : z>{cls.STEP_ARTIFACT_Z} over "
            f"{cls.STEP_ARTIFACT_WINDOW_MS} ms windows",
            f"  Min valid leads     : {cls.MIN_VALID_LEADS}/{cls.NUM_LEADS}",
            f"  R-peak lead (II/V1) : idx {cls.LEAD_II_IDX} / idx {cls.LEAD_V1_IDX}",
            f"  Beat window         : {cls.PRE_R_MS} ms pre / {cls.POST_R_MS} ms post",
            f"  Min beats/record    : {cls.MIN_BEATS_PER_RECORD}",
            f"  Num classes         : {cls.NUM_CLASSES}  {cls.CLASS_NAMES}",
            f"  Val / test split    : {cls.VALIDATION_SPLIT} / {cls.TEST_SPLIT}",
            f"  Data path           : {cls.DATA_PATH}",
            f"  Processed path      : {cls.PROCESSED_PATH}",
            "=" * 55,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    PhysioNetConfig.create_output_dirs()
    PhysioNetConfig.load_label_vocab()
    print(PhysioNetConfig.summary())