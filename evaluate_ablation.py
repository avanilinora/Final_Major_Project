"""
evaluate_ablation.py
=====================
Evaluates all 4 ablation variants using IDENTICAL evaluation code — same
metrics, same label range, same test set — so the numbers in your
ablation table are a fair, apples-to-apples comparison, not an artifact
of using slightly different evaluation logic per variant.

Usage
-----
    # Evaluate one variant
    python evaluate_ablation.py --variant no_lead_attention

    # Evaluate all 4 and print the final table (run this once all checkpoints exist)
    python evaluate_ablation.py --variant all
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config_physionet import PhysioNetConfig as PNConfig
from models.cnn2 import CNNBaseline
from models.cnn_bilstm2 import CNNBiLSTM
from utils.dataset2 import build_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

VARIANTS = {
    "full_model": {
        "build": lambda: CNNBiLSTM(num_classes=PNConfig.NUM_CLASSES, num_leads=PNConfig.NUM_LEADS,
                                    use_lead_attention=True, use_temporal_attention=True),
        "checkpoint": "physionet_cnn_bilstm2_best_model.pth",
        "label": "Full model (CNN+LeadAttn+BiLSTM+TemporalAttn)",
    },
    "cnn_only": {
        "build": lambda: CNNBaseline(num_classes=PNConfig.NUM_CLASSES, num_leads=PNConfig.NUM_LEADS,
                                     use_lead_attention=True),
        "checkpoint": "physionet_cnn2_best_model.pth",
        "label": "CNN only (no BiLSTM, no temporal attention)",
    },
    "no_lead_attention": {
        "build": lambda: CNNBiLSTM(num_classes=PNConfig.NUM_CLASSES, num_leads=PNConfig.NUM_LEADS,
                                    use_lead_attention=False, use_temporal_attention=True),
        "checkpoint": "ablation_no_lead_attention_best_model.pth",
        "label": "No lead attention (mean-pool leads)",
    },
    "no_temporal_attention": {
        "build": lambda: CNNBiLSTM(num_classes=PNConfig.NUM_CLASSES, num_leads=PNConfig.NUM_LEADS,
                                    use_lead_attention=True, use_temporal_attention=False),
        "checkpoint": "ablation_no_temporal_attention_best_model.pth",
        "label": "No temporal attention (final hidden state)",
    },
}


@torch.no_grad()
def evaluate_one(variant_key: str, test_loader, device: torch.device) -> dict:
    spec = VARIANTS[variant_key]
    checkpoint_path = os.path.join(PNConfig.CHECKPOINT_PATH, spec["checkpoint"])
    if not os.path.exists(checkpoint_path):
        logger.warning("Skipping '%s' — checkpoint not found: %s", variant_key, checkpoint_path)
        return None

    model = spec["build"]()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    all_preds, all_labels = [], []
    for samples, labels in test_loader:
        samples = samples.to(device)
        output = model(samples)
        logits = output[0] if isinstance(output, tuple) else output
        preds = F.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    y_pred = np.array(all_preds)
    y_test = np.array(all_labels)
    labels_list = list(range(PNConfig.NUM_CLASSES))

    return {
        "variant": variant_key,
        "label": spec["label"],
        "checkpoint": spec["checkpoint"],
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="macro", labels=labels_list, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="macro", labels=labels_list, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="macro", labels=labels_list, zero_division=0)),
    }


def print_table(results: list) -> None:
    full = next((r for r in results if r["variant"] == "full_model"), None)
    print("\n" + "=" * 100)
    print(f"{'Variant':<45}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}{'DeltaAcc':>10}")
    print("=" * 100)
    for r in results:
        delta = (r["accuracy"] - full["accuracy"]) if full else 0.0
        delta_str = f"{delta:+.4f}" if full else "-"
        print(f"{r['label']:<45}{r['accuracy']:>10.4f}{r['precision']:>11.4f}"
              f"{r['recall']:>9.4f}{r['f1']:>9.4f}{delta_str:>10}")
    print("=" * 100)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one or all ablation variants with identical evaluation code.")
    parser.add_argument("--variant", type=str, default="all", choices=list(VARIANTS.keys()) + ["all"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_json", type=str, default="ablation_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    PNConfig.load_label_vocab()

    _, _, test_loader = build_dataloaders(
        processed_path=PNConfig.PROCESSED_PATH,
        batch_size=args.batch_size,
        num_workers=0,
        use_weighted_sampler=False,
        num_classes=PNConfig.NUM_CLASSES,
    )

    variants_to_run = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]

    results = []
    for v in variants_to_run:
        logger.info("Evaluating: %s", v)
        r = evaluate_one(v, test_loader, PNConfig.DEVICE)
        if r is not None:
            results.append(r)
            logger.info("  acc=%.4f  macro-F1=%.4f", r["accuracy"], r["f1"])

    if not results:
        logger.error("No checkpoints found for any requested variant — nothing to report.")
        return

    print_table(results)

    with open(args.output_json, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Full results saved -> %s", args.output_json)


if __name__ == "__main__":
    main()