"""
evaluate_physionet.py
======================
Test-set evaluation for the PhysioNet 12-lead ECG-Arrhythmia track.

*** FIX applied ***
`calculate_all_metrics()` and `plot_confusion_matrix()` are now called
with explicit `num_classes=PNConfig.NUM_CLASSES` / `class_names=PNConfig.CLASS_NAMES`
so the utils/metrics.py and utils/visualization.py fixes actually take
effect here too (previously called with no override, so it would have
silently fallen back to whatever labels happened to be present, or in
the original buggy version, hardcoded to 4 classes).

Usage
-----
    python evaluate_physionet.py --model cnn2
    python evaluate_physionet.py --model cnn_bilstm2 --checkpoint path/to/model.pth
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score

from config_physionet import PhysioNetConfig as PNConfig
from train_physionet import MODEL_REGISTRY, build_model
from trainer import set_seed

from utils.dataset2 import build_dataloaders
from utils.metrics import calculate_all_metrics
from utils.visualization import plot_confusion_matrix, plot_roc_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> nn.Module:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train the model first with: python train_physionet.py --model <name>"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded checkpoint: %s (epoch=%s)", checkpoint_path, checkpoint.get("epoch", "?"))
    return model


def run_evaluation(model: nn.Module, model_name: str, test_loader, device) -> Dict[str, float]:
    """Full test-set evaluation."""
    model = model.to(device)
    model.eval()

    all_preds, all_labels, all_probs, inference_times = [], [], [], []

    with torch.no_grad():
        for samples, labels in test_loader:
            samples = samples.to(device)

            t0 = time.perf_counter()
            output = model(samples)
            t1 = time.perf_counter()
            inference_times.append((t1 - t0) / len(labels))

            logits = output[0] if isinstance(output, tuple) else output
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.append(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.vstack(all_probs)

    metrics = calculate_all_metrics(
        y_true, y_pred, average="macro", num_classes=PNConfig.NUM_CLASSES
    )

    try:
        roc_auc = roc_auc_score(
            y_true, y_prob, average="macro", multi_class="ovr",
            labels=list(range(PNConfig.NUM_CLASSES)),
        )
    except ValueError:
        roc_auc = float("nan")
    metrics["roc_auc"] = roc_auc

    avg_inference_ms = float(np.mean(inference_times)) * 1000

    logger.info("-" * 50)
    logger.info("  Model            : %s", model_name)
    logger.info("  Accuracy         : %.4f", metrics["accuracy"])
    logger.info("  Precision (macro): %.4f", metrics["precision"])
    logger.info("  Recall    (macro): %.4f", metrics["recall"])
    logger.info("  F1        (macro): %.4f", metrics["f1"])
    logger.info("  ROC-AUC   (macro): %.4f", roc_auc)
    logger.info("  Inf. time        : %.3f ms/sample", avg_inference_ms)
    logger.info("-" * 50)

    report = classification_report(
        y_true, y_pred, labels=list(range(PNConfig.NUM_CLASSES)),
        target_names=PNConfig.CLASS_NAMES, digits=4, zero_division=0,
    )
    print("\n" + report)

    os.makedirs(PNConfig.PLOTS_PATH, exist_ok=True)
    report_path = os.path.join(PNConfig.PLOTS_PATH, f"{model_name}_classification_report.txt")
    with open(report_path, "w") as fh:
        fh.write(f"Model: {model_name}\n\n")
        for k, v in metrics.items():
            fh.write(f"{k}: {v:.6f}\n")
        fh.write(f"inference_ms_per_sample: {avg_inference_ms:.4f}\n\n")
        fh.write(report)
    logger.info("Classification report saved -> %s", report_path)

    plot_confusion_matrix(
        y_true, y_pred, model_name=model_name,
        class_names=PNConfig.CLASS_NAMES,
        normalise=True, save_dir=PNConfig.CONFUSION_MATRIX_PATH,
    )
    plot_confusion_matrix(
        y_true, y_pred, model_name=model_name,
        class_names=PNConfig.CLASS_NAMES,
        normalise=False, save_dir=PNConfig.CONFUSION_MATRIX_PATH,
    )
    plot_roc_curve(
        y_true, y_prob, model_name=model_name,
        class_names=PNConfig.CLASS_NAMES, save_dir=PNConfig.PLOTS_PATH,
    )

    logger.info("Evaluation complete.")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PhysioNet 12-lead ECG-Arrhythmia — Evaluation Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=PNConfig.BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=PNConfig.RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PNConfig.create_output_dirs()
    PNConfig.load_label_vocab()
    set_seed(args.seed)

    _train_loader, _val_loader, test_loader = build_dataloaders(
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

    run_evaluation(model, f"physionet_{args.model}", test_loader, PNConfig.DEVICE)


if __name__ == "__main__":
    main()
