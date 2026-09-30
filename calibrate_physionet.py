"""
calibrate_physionet.py
=======================
Day 6 — confidence calibration.

Measures whether the trained model's stated confidence can be trusted
(reliability diagram + Expected Calibration Error), then fixes it via
temperature scaling (utils/calibration.py) fit on the validation set, and
re-measures on the test set to show the before/after effect.

Does NOT retrain the model and does NOT change its accuracy — temperature
scaling only reshapes confidence, not predictions (see calibration.py's
module docstring for why).

Usage
-----
    python calibrate_physionet.py --model cnn_bilstm2
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import torch

from config_physionet import PhysioNetConfig as PNConfig
from train_physionet import MODEL_REGISTRY, build_model
from trainer import set_seed
from utils.calibration import fit_temperature, get_calibrated_probs
from utils.dataset2 import build_dataloaders
from utils.visualization import plot_reliability_diagram

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def load_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train it first: python train_physionet.py --model <name>"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded checkpoint: %s (epoch=%s)", checkpoint_path, checkpoint.get("epoch", "?"))
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Confidence calibration via temperature scaling.")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=PNConfig.BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=PNConfig.RANDOM_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    PNConfig.create_output_dirs()
    PNConfig.load_label_vocab()
    set_seed(args.seed)

    _train_loader, val_loader, test_loader = build_dataloaders(
        processed_path=PNConfig.PROCESSED_PATH,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=False,
        num_classes=PNConfig.NUM_CLASSES,
    )

    model = build_model(args.model)
    checkpoint_path = args.checkpoint or os.path.join(
        PNConfig.CHECKPOINT_PATH, f"physionet_{args.model}_best_model.pth"
    )
    model = load_checkpoint(model, checkpoint_path, PNConfig.DEVICE)
    model_name = f"physionet_{args.model}"

    # ------------------------------------------------------------------
    # 1. BEFORE — raw, uncalibrated confidence on the test set
    # ------------------------------------------------------------------
    logger.info("=" * 55)
    logger.info("Measuring uncalibrated (raw) confidence on the test set...")
    y_true, y_pred, probs_before = get_calibrated_probs(model, test_loader, PNConfig.DEVICE, temperature=1.0)
    confidences_before = probs_before.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    accuracy = float(correct.mean())

    _, ece_before = plot_reliability_diagram(
        confidences_before, correct, model_name=model_name,
        save_dir=PNConfig.PLOTS_PATH, title_suffix=" (before)",
    )
    logger.info("Accuracy (unaffected by calibration): %.4f", accuracy)
    logger.info("ECE before calibration: %.4f", ece_before)

    # ------------------------------------------------------------------
    # 2. Fit temperature on the VALIDATION set (never test)
    # ------------------------------------------------------------------
    logger.info("=" * 55)
    logger.info("Fitting temperature on the validation set...")
    T = fit_temperature(model, val_loader, PNConfig.DEVICE)

    # ------------------------------------------------------------------
    # 3. AFTER — calibrated confidence on the test set
    # ------------------------------------------------------------------
    logger.info("=" * 55)
    logger.info("Measuring calibrated confidence on the test set...")
    y_true_after, y_pred_after, probs_after = get_calibrated_probs(
        model, test_loader, PNConfig.DEVICE, temperature=T
    )
    confidences_after = probs_after.max(axis=1)
    correct_after = (y_pred_after == y_true_after).astype(float)
    accuracy_after = float(correct_after.mean())

    _, ece_after = plot_reliability_diagram(
        confidences_after, correct_after, model_name=model_name,
        save_dir=PNConfig.PLOTS_PATH, title_suffix=" (after)",
    )

    # Sanity guarantee: temperature scaling must not change predictions.
    assert np.array_equal(y_pred, y_pred_after), (
        "Predictions changed after temperature scaling — this should be "
        "mathematically impossible and indicates a bug, not a calibration "
        "result. Do not trust these numbers if you see this."
    )
    assert abs(accuracy - accuracy_after) < 1e-9, "Accuracy changed — see above."

    logger.info("Accuracy after calibration: %.4f  (must match before: %.4f)", accuracy_after, accuracy)
    logger.info("ECE after calibration: %.4f", ece_after)
    logger.info("=" * 55)
    logger.info("SUMMARY")
    logger.info("  Temperature (T)     : %.4f", T)
    logger.info("  Accuracy            : %.4f  (unchanged, as expected)", accuracy)
    logger.info("  ECE before          : %.4f", ece_before)
    logger.info("  ECE after           : %.4f", ece_after)
    improvement = (ece_before - ece_after) / ece_before * 100 if ece_before > 0 else 0.0
    logger.info("  ECE improvement     : %.1f%%", improvement)
    logger.info("=" * 55)

    # ------------------------------------------------------------------
    # Save the fitted temperature for reuse in inference/the web app
    # ------------------------------------------------------------------
    calib_path = os.path.join(PNConfig.CHECKPOINT_PATH, f"{model_name}_temperature.json")
    with open(calib_path, "w") as fh:
        json.dump({
            "model": model_name, "temperature": T,
            "accuracy": accuracy, "ece_before": ece_before, "ece_after": ece_after,
        }, fh, indent=2)
    logger.info("Saved calibration result -> %s", calib_path)
    logger.info("Use this by dividing logits by T before softmax at inference time.")


if __name__ == "__main__":
    main()