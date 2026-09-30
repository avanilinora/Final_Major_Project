"""
train_ablation.py
==================
Trains the two ablation variants not already covered by existing
checkpoints:
  - "no_lead_attention"     : mean-pool across leads instead of learned attention
  - "no_temporal_attention" : use the BiLSTM's final hidden state instead of
                               attention-weighted pooling over time

Your existing checkpoints already cover the other two rows of the
ablation table:
  - "full_model"  = outputs/checkpoints_physionet/physionet_cnn_bilstm2_best_model.pth
  - "cnn_only"    = outputs/checkpoints_physionet/physionet_cnn2_best_model.pth

Usage
-----
    python train_ablation.py --variant no_lead_attention
    python train_ablation.py --variant no_temporal_attention
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import torch
from torch import Tensor

from config_physionet import PhysioNetConfig as PNConfig
from models.cnn_bilstm2 import CNNBiLSTM
from trainer import BaseTrainer, set_seed
from utils.dataset2 import ECGDataset, build_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VARIANTS = {
    "no_lead_attention":     dict(use_lead_attention=False, use_temporal_attention=True),
    "no_temporal_attention": dict(use_lead_attention=True,  use_temporal_attention=False),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train an ablation variant of CNNBiLSTM.")
    parser.add_argument("--variant", type=str, required=True, choices=list(VARIANTS.keys()))
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


def main():
    args = parse_args()
    PNConfig.create_output_dirs()
    PNConfig.load_label_vocab()
    set_seed(args.seed)

    logger.info("Training ablation variant: %s  (%s)", args.variant, VARIANTS[args.variant])

    train_loader, val_loader, _test_loader = build_dataloaders(
        processed_path=PNConfig.PROCESSED_PATH,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=not args.no_weighted_sampler,  # Match train_physionet default
        num_classes=PNConfig.NUM_CLASSES,
    )

    class_weights: Optional[Tensor] = None
    if not args.no_class_weights:
        train_ds: ECGDataset = train_loader.dataset  # type: ignore[assignment]
        class_weights = train_ds.class_weights
        logger.info("Class weights: %s", class_weights.tolist())

    model = CNNBiLSTM(
        num_classes=PNConfig.NUM_CLASSES,
        num_leads=PNConfig.NUM_LEADS,
        **VARIANTS[args.variant],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")

    model_name = f"ablation_{args.variant}"
    trainer = BaseTrainer(
        model=model,
        model_name=model_name,
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
        num_classes=PNConfig.NUM_CLASSES,
    )
    trainer.train()

    logger.info("Done. Checkpoint saved as: %s_best_model.pth", model_name)


if __name__ == "__main__":
    main()