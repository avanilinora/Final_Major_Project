"""
export_probs.py
================
Exports YOUR model's val/test softmax probabilities as .npy files, in the
exact format ensemble_models.py expects. Run this once you have a final
checkpoint — the output is what you hand off for ensembling.

Usage
-----
    python export_probs.py --model cnn_bilstm2
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from config_physionet import PhysioNetConfig as PNConfig
from models.cnn2 import CNNBaseline
from models.cnn_bilstm2 import CNNBiLSTM
from utils.dataset2 import build_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "cnn2": CNNBaseline,
    "cnn_bilstm2": CNNBiLSTM,
}


@torch.no_grad()
def export_loader(model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []

    for samples, labels in loader:
        samples = samples.to(device)
        output = model(samples)
        
        # FIX 1: Safely unpack tuple output from CNNBiLSTM
        logits = output[0] if isinstance(output, tuple) else output
        
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    return np.vstack(all_probs), np.concatenate(all_labels)


def parse_args():
    parser = argparse.ArgumentParser(description="Export val/test probabilities for ensembling.")
    parser.add_argument("--model", type=str, default="cnn_bilstm2", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model_tag", type=str, default="cnn_bilstm",
                        help="Short name used in the output filenames (e.g. val_probs_<tag>.npy).")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    PNConfig.load_label_vocab()

    # FIX 2: Use build_dataloaders to ensure consistent dataset transformation pipeline
    _, val_loader, test_loader = build_dataloaders(
        processed_path=PNConfig.PROCESSED_PATH,
        batch_size=args.batch_size,
        num_workers=0,
        use_weighted_sampler=False,
        num_classes=PNConfig.NUM_CLASSES,
    )

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(num_classes=PNConfig.NUM_CLASSES, num_leads=PNConfig.NUM_LEADS)

    checkpoint_path = args.checkpoint or os.path.join(
        PNConfig.CHECKPOINT_PATH, f"physionet_{args.model}_best_model.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location=PNConfig.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(PNConfig.DEVICE)
    logger.info("Loaded checkpoint: %s", checkpoint_path)

    val_probs, y_val = export_loader(model, val_loader, PNConfig.DEVICE)
    test_probs, y_test = export_loader(model, test_loader, PNConfig.DEVICE)

    os.makedirs(args.output_dir, exist_ok=True)
    val_path = os.path.join(args.output_dir, f"val_probs_{args.model_tag}.npy")
    test_path = os.path.join(args.output_dir, f"test_probs_{args.model_tag}.npy")
    yval_path = os.path.join(args.output_dir, "y_val.npy")
    ytest_path = os.path.join(args.output_dir, "y_test.npy")

    np.save(val_path, val_probs)
    np.save(test_path, test_probs)
    np.save(yval_path, y_val)
    np.save(ytest_path, y_test)

    logger.info("Saved: %s  shape=%s", val_path, val_probs.shape)
    logger.info("Saved: %s  shape=%s", test_path, test_probs.shape)
    logger.info("Saved: %s, %s", yval_path, ytest_path)
    logger.info("Sanity check: rows sum to 1.0? %s", np.allclose(val_probs.sum(axis=1), 1.0))


if __name__ == "__main__":
    main()