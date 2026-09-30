"""
explain_user_recording.py
===========================
The user-facing entry point: give this ONE ECG recording, get a full
XAI Word report back, in one command.

Accepts either:
  --input path/to/record          (a raw WFDB record, e.g. .../JS00001 —
                                    no extension; .hea/.mat must exist
                                    alongside it) — this is run through
                                    the EXACT SAME preprocessing as
                                    training (band-pass, wavelet denoise,
                                    per-lead z-score, fixed-length window),
                                    so the model sees data in the same
                                    distribution it was trained on.
  --input path/to/array.npy       (an already-preprocessed array, shape
                                    (5000, 12) — e.g. one row pulled out
                                    of X_test.npy for testing this script)

Usage
-----
    python explain_user_recording.py --model cnn_bilstm2 --input data/Physionet_dataset/WFDBRecords/01/010/JS00001
    python explain_user_recording.py --model cnn_bilstm2 --input my_recording.npy
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import torch

from build_xai_report import build_single_recording_report
from config_physionet import PhysioNetConfig as PNConfig
from explain_recording import load_checkpoint
from train_physionet import MODEL_REGISTRY, build_model
from trainer import set_seed
from xai.pipeline import generate_assets_for_recording

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

NORMAL_CLASS_NAME = "SR"


def load_recording(input_path: str) -> np.ndarray:
    """
    Returns a preprocessed (5000, 12) array, ready for the model.
    """
    if input_path.endswith(".npy"):
        x_np = np.load(input_path)
        if x_np.shape != (PNConfig.RECORD_LENGTH, PNConfig.NUM_LEADS):
            raise ValueError(
                f"Expected a preprocessed array of shape "
                f"({PNConfig.RECORD_LENGTH}, {PNConfig.NUM_LEADS}), got {x_np.shape}. "
                f"If this is a RAW recording, pass the WFDB record path instead "
                f"(no .npy extension) so it gets preprocessed automatically."
            )
        logger.info("Loaded pre-processed recording from %s", input_path)
        return x_np.astype(np.float32)

    # Otherwise: treat as a raw WFDB record and run the SAME preprocessing
    # pipeline used during training, so the model sees matching-distribution
    # input rather than raw, unfiltered signal.
    from utils.preprocessing_physionet import (
        bandpass_filter, fix_length, load_signal, normalize_signal, wavelet_denoise,
    )

    logger.info("Loading raw WFDB record: %s", input_path)
    signal = load_signal(input_path)
    signal = bandpass_filter(signal)
    signal = wavelet_denoise(signal)
    signal = normalize_signal(signal)
    signal = fix_length(signal)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    logger.info("Preprocessed raw recording to shape %s", signal.shape)
    return signal.astype(np.float32)


def load_normal_reference():
    """
    Best-effort: use a real Sinus Rhythm example from the processed test
    set as the "normal" comparison reference. Returns None (comparison
    figures are then skipped, gracefully) if processed test data isn't
    available — e.g. in a deployment that ships only the model.
    """
    try:
        X_test = np.load(os.path.join(PNConfig.PROCESSED_PATH, "X_test.npy"))
        y_test = np.load(os.path.join(PNConfig.PROCESSED_PATH, "y_test.npy"))
        normal_idx = PNConfig.CLASS_NAMES.index(NORMAL_CLASS_NAME)
        matches = np.where(y_test == normal_idx)[0]
        if len(matches) == 0:
            logger.warning("No Sinus Rhythm examples found in test set — skipping normal-comparison figures.")
            return None
        logger.info("Using test-set Sinus Rhythm example (index %d) as the normal reference.", matches[0])
        return X_test[matches[0]]
    except FileNotFoundError:
        logger.warning(
            "No processed test set found at %s — skipping normal-comparison figures. "
            "(Run utils/preprocessing_physionet.py to enable this.)",
            PNConfig.PROCESSED_PATH,
        )
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Explain one user-provided ECG recording, end to end.")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--input", type=str, required=True, help="Path to a raw WFDB record (no extension) or a preprocessed .npy array.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="Where to save figures + report. Defaults to outputs/plots_physionet/xai_report_user/<input filename>.")
    parser.add_argument("--seed", type=int, default=PNConfig.RANDOM_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    PNConfig.create_output_dirs()
    PNConfig.load_label_vocab()
    set_seed(args.seed)

    class_names = PNConfig.CLASS_NAMES
    lead_names = PNConfig.LEAD_NAMES
    fs = PNConfig.SAMPLING_FREQ

    x_np = load_recording(args.input)
    normal_signal_full = load_normal_reference()

    model = build_model(args.model)
    checkpoint_path = args.checkpoint or os.path.join(
        PNConfig.CHECKPOINT_PATH, f"physionet_{args.model}_best_model.pth"
    )
    model = load_checkpoint(model, checkpoint_path, PNConfig.DEVICE)
    model = model.to(PNConfig.DEVICE).eval()

    recording_name = os.path.splitext(os.path.basename(args.input))[0]
    save_dir = args.output_dir or os.path.join(PNConfig.PLOTS_PATH, "xai_report_user", recording_name)

    logger.info("Running explanation pipeline for: %s", args.input)
    recording_summary = generate_assets_for_recording(
        model, x_np, None, save_dir, class_names, lead_names, fs, PNConfig.DEVICE,
        normal_signal_full=normal_signal_full, normal_class_name=NORMAL_CLASS_NAME,
    )

    logger.info(
        "Predicted: %s  (confidence %.1f%%)",
        recording_summary["predicted_class"], recording_summary["confidence"] * 100,
    )

    summary = {
        "mode": "single_recording",
        "model": args.model,
        "input": args.input,
        "normal_reference": {"class": NORMAL_CLASS_NAME} if normal_signal_full is not None else None,
        "recording": recording_summary,
    }
    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Summary -> %s", summary_path)

    report_path = os.path.join(save_dir, "TrustECG_Explanation_Report.docx")
    build_single_recording_report(summary, save_dir, report_path)
    logger.info("=" * 55)
    logger.info("Report ready -> %s", report_path)
    logger.info("Full per-lead figures -> %s", save_dir)


if __name__ == "__main__":
    main()