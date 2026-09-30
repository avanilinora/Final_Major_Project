"""
train_physionet.py
===================
Training entry-point for the PhysioNet 12-lead ECG-Arrhythmia
(record-level rhythm/diagnosis) track.

Reuses the shared `BaseTrainer`, `EarlyStopping`, `set_seed` from
trainer.py — only the config, data loading, and model construction differ
from the MIT-BIH track.

Usage
-----
    python train_physionet.py --model cnn2
    python train_physionet.py --model cnn_bilstm2 --epochs 50 --batch_size 32

Prerequisite
------------
    python utils/preprocessing_physionet.py
"""

from __future__ import annotations

import argparse
import logging
from typing import Dict, Optional

import numpy as np
import torch.nn as nn
from torch import Tensor

from config_physionet import PhysioNetConfig as PNConfig
from models.cnn2 import CNNBaseline
from models.cnn_bilstm2 import CNNBiLSTM
from trainer import BaseTrainer, set_seed
from utils.dataset2 import ECGDataset, build_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Model registry — ALWAYS injects num_leads/num_classes explicitly
# ===========================================================================

MODEL_REGISTRY: Dict[str, type] = {
    "cnn2": CNNBaseline,
    "cnn_bilstm2": CNNBiLSTM,
}


def build_model(model_name: str) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](
        num_classes=PNConfig.NUM_CLASSES,
        num_leads=PNConfig.NUM_LEADS,
    )


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PhysioNet 12-lead ECG-Arrhythmia — Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True,
                         choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--epochs", type=int, default=PNConfig.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=PNConfig.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=PNConfig.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=PNConfig.WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=PNConfig.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--no_weighted_sampler", action="store_true", default=False)
    parser.add_argument("--no_class_weights", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=PNConfig.RANDOM_SEED)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()

    PNConfig.create_output_dirs()
    PNConfig.load_label_vocab()   # must run preprocessing_physionet.py first
    set_seed(args.seed)

    logger.info(PNConfig.summary())
    logger.info("Selected model  : %s", args.model)

    train_loader, val_loader, _test_loader = build_dataloaders(
        processed_path=PNConfig.PROCESSED_PATH,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=not args.no_weighted_sampler,
        num_classes=PNConfig.NUM_CLASSES,
    )

    class_weights: Optional[Tensor] = None
    if not args.no_class_weights:
        train_ds: ECGDataset = train_loader.dataset  # type: ignore[assignment]
        class_weights = train_ds.class_weights
        logger.info("Class weights: %s", class_weights.tolist())

    model = build_model(args.model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")

    trainer = BaseTrainer(
        model=model,
        model_name=f"physionet_{args.model}",
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        patience=args.patience,
        device=PNConfig.DEVICE,
        checkpoint_dir=PNConfig.CHECKPOINT_PATH,
        plots_dir=PNConfig.PLOTS_PATH,
        num_classes=PNConfig.NUM_CLASSES,   # <-- the metrics.py fix, wired in
    )

    history = trainer.train()

    best_epoch = int(np.argmax(history["val_f1"])) + 1
    logger.info("Best epoch : %d  (val_f1=%.4f)", best_epoch, history["val_f1"][best_epoch - 1])
    logger.info("Run `python evaluate_physionet.py --model %s` to evaluate on the test set.",
                args.model)


if __name__ == "__main__":
    main()
