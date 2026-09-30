"""
generate_xai_report_assets.py
==============================
Generates the "one representative example per class" XAI report assets —
uses xai/pipeline.py's shared generate_assets_for_recording(), the same
routine explain_user_recording.py uses for a single new recording.

    outputs/plots_physionet/xai_report/
      {class_name}/
        lead_attention.png
        twelve_lead_grid.png
        lead_01_I/ ... lead_12_V6/
          technical.png, gradcam_saliency.png, temporal_attention.png,
          vs_normal.png (omitted for the Normal/SR class itself)

Selection strategy: for each class, the FIRST correctly-classified test
example of that class is used (falls back to the first example of that
class at all if none are classified correctly — reported honestly in
the summary JSON either way).

Usage
-----
    python generate_xai_report_assets.py --model cnn_bilstm2
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import torch

from config_physionet import PhysioNetConfig as PNConfig
from explain_recording import load_checkpoint
from train_physionet import MODEL_REGISTRY, build_model
from trainer import set_seed
from xai.pipeline import generate_assets_for_recording

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

NORMAL_CLASS_NAME = "SR"  # Sinus Rhythm — used as the "normal" reference throughout


def select_representative_index(y_test: np.ndarray, y_pred: np.ndarray, class_idx: int) -> tuple[int, bool]:
    class_mask = y_test == class_idx
    correct_mask = class_mask & (y_pred == y_test)
    if correct_mask.any():
        return int(np.where(correct_mask)[0][0]), True
    if class_mask.any():
        return int(np.where(class_mask)[0][0]), False
    raise ValueError(f"No test examples found for class index {class_idx}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate all XAI report assets, for every class and every lead.")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=PNConfig.RANDOM_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    PNConfig.create_output_dirs()
    PNConfig.load_label_vocab()
    set_seed(args.seed)

    X_test = np.load(os.path.join(PNConfig.PROCESSED_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(PNConfig.PROCESSED_PATH, "y_test.npy"))
    class_names = PNConfig.CLASS_NAMES
    lead_names = PNConfig.LEAD_NAMES
    fs = PNConfig.SAMPLING_FREQ

    model = build_model(args.model)
    checkpoint_path = args.checkpoint or os.path.join(
        PNConfig.CHECKPOINT_PATH, f"physionet_{args.model}_best_model.pth"
    )
    model = load_checkpoint(model, checkpoint_path, PNConfig.DEVICE)
    model = model.to(PNConfig.DEVICE).eval()

    logger.info("Running inference on the full test set for example selection...")
    y_pred = np.zeros_like(y_test)
    with torch.no_grad():
        for i in range(len(X_test)):
            x = torch.from_numpy(X_test[i]).float().unsqueeze(0).to(PNConfig.DEVICE)
            logits = model(x)
            y_pred[i] = int(logits.argmax(dim=1).item())

    report_dir = os.path.join(PNConfig.PLOTS_PATH, "xai_report")
    os.makedirs(report_dir, exist_ok=True)

    normal_class_idx = class_names.index(NORMAL_CLASS_NAME)
    normal_test_idx, normal_correct = select_representative_index(y_test, y_pred, normal_class_idx)
    normal_signal_full = X_test[normal_test_idx]

    summary = {
        "mode": "batch_per_class",
        "model": args.model,
        "normal_reference": {"class": NORMAL_CLASS_NAME, "test_index": normal_test_idx, "was_correctly_classified": normal_correct},
        "classes": {},
    }

    for class_idx, class_name in enumerate(class_names):
        logger.info("=" * 55)
        logger.info("Class: %s", class_name)

        test_idx, was_correct = select_representative_index(y_test, y_pred, class_idx)
        x_np = X_test[test_idx]

        logger.info("  Selected test index %d (correctly classified: %s)", test_idx, was_correct)

        class_dir = os.path.join(report_dir, class_name)
        class_summary = generate_assets_for_recording(
            model, x_np, class_name, class_dir, class_names, lead_names, fs, PNConfig.DEVICE,
            normal_signal_full=normal_signal_full, normal_class_name=NORMAL_CLASS_NAME,
        )
        class_summary["test_index"] = test_idx
        class_summary["was_correctly_classified"] = was_correct

        logger.info("  Predicted: %s  (confidence %.1f%%)", class_summary["predicted_class"], class_summary["confidence"] * 100)
        logger.info("  Generated all 12 lead folders under %s", class_dir)
        summary["classes"][class_name] = class_summary

    summary_path = os.path.join(report_dir, "xai_report_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("=" * 55)
    logger.info("All assets generated -> %s", report_dir)
    logger.info("Summary -> %s", summary_path)
    logger.info("Next: python build_xai_report.py --model %s", args.model)


if __name__ == "__main__":
    main()