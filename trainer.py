"""
trainer.py  (renamed from train.py)
====================================
Shared training pipeline used by BOTH tracks: `BaseTrainer`, `EarlyStopping`,
`set_seed`. train_physionet.py imports these directly.

*** CHANGE ***
`BaseTrainer` now accepts an optional `num_classes` argument and threads it
through to `calculate_all_metrics()`. This is required for the metrics.py
fix (see utils/metrics.py docstring) to actually take effect during
training — without it, BaseTrainer would keep silently defaulting to
Phase-1's 4-class label range even after the metrics functions themselves
were fixed. train_physionet.py passes `num_classes=PNConfig.NUM_CLASSES`.
Existing MIT-BIH call sites are unaffected (default: None -> metrics infer
labels from the data actually seen, which for 4-class data is equivalent).

*** SPEED FIX: mixed precision (AMP) added ***
Training now runs the forward pass + loss under `torch.autocast` and
scales gradients with `torch.cuda.amp.GradScaler`. On a CUDA GPU this
typically gives a real speedup (often 1.5-3x) with no accuracy cost — it
runs most ops in float16 while keeping a float32 master copy of weights,
so precision-sensitive steps (loss scaling, weight updates) stay safe.
On CPU, `torch.autocast(device_type=..., enabled=(device.type=="cuda"))`
means this is automatically a no-op — safe either way, no separate CPU
code path needed. Only the training loop is touched; validation still
runs a plain forward pass (no gradient, no need for AMP there), so
metrics/checkpoint selection are unaffected in precision.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config_physionet import Config
from utils.metrics import calculate_all_metrics
from utils.visualization import plot_accuracy_curve, plot_loss_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Reproducibility
# ===========================================================================

def set_seed(seed: int = Config.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ===========================================================================
# Early stopping helper
# ===========================================================================

class EarlyStopping:

    def __init__(
        self,
        patience: int = Config.EARLY_STOPPING_PATIENCE,
        min_delta: float = 1e-5,
        mode: str = "min",
        verbose: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter: int = 0
        self.best_value: float = float("inf") if mode == "min" else -float("inf")
        self.should_stop: bool = False

    def __call__(self, current_value: float) -> bool:
        improved = (
            (current_value < self.best_value - self.min_delta)
            if self.mode == "min"
            else (current_value > self.best_value + self.min_delta)
        )

        if improved:
            self.best_value = current_value
            self.counter = 0
            if self.verbose:
                logger.info("  EarlyStopping: new best = %.6f", self.best_value)
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    "  EarlyStopping: no improvement (%d / %d)",
                    self.counter, self.patience,
                )
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ===========================================================================
# BaseTrainer — shared training pipeline
# ===========================================================================

class BaseTrainer:
    """
    Shared training and validation loop for all TrustECG models.

    * AdamW optimiser
    * CrossEntropyLoss with optional class-weighting
    * ReduceLROnPlateau learning-rate scheduler
    * Early stopping on validation macro-F1
    * Best-checkpoint saving
    * Automatic curve saving via utils.visualization
    * Mixed precision (AMP) training on CUDA, no-op on CPU
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[Tensor] = None,
        learning_rate: float = Config.LEARNING_RATE,
        weight_decay: float = Config.WEIGHT_DECAY,
        num_epochs: int = Config.NUM_EPOCHS,
        patience: int = Config.EARLY_STOPPING_PATIENCE,
        device: torch.device = Config.DEVICE,
        checkpoint_dir: str = Config.CHECKPOINT_PATH,
        plots_dir: str = Config.PLOTS_PATH,
        num_classes: Optional[int] = None,
        use_amp: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.plots_dir = plots_dir
        self.num_classes = num_classes

        # AMP only actually engages on CUDA — on CPU this is a safe no-op,
        # so `use_amp=True` is fine to leave as the default regardless of
        # what device you're on.
        self.use_amp = use_amp and device.type == "cuda"
        try:
            # torch>=2.4 API
            self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_amp)
        except TypeError:
            # older torch — the FutureWarning is harmless either way, this
            # just avoids it on newer versions
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if use_amp and device.type != "cuda":
            logger.info("AMP requested but device is CPU — running in full precision.")
        elif self.use_amp:
            logger.info("Mixed precision (AMP) training enabled.")

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        if class_weights is not None:
            class_weights = class_weights.to(device)
            logger.info("Using class weights: %s", class_weights.cpu().numpy())
        else:
            logger.info("Using uniform class weights.")

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.1,
        )

        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )

        self.early_stopping = EarlyStopping(patience=patience, mode="max")

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
            "val_f1": [], "val_precision": [], "val_recall": [],
        }

        self.best_val_f1: float = 0.0
        self.best_checkpoint_path: str = os.path.join(
            checkpoint_dir, f"{model_name}_best_model.pth"
        )

    # ------------------------------------------------------------------

    def _train_one_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []

        for beats, labels in self.train_loader:
            beats = beats.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                output = self.model(beats)
                logits = output[0] if isinstance(output, tuple) else output
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            # Unscale before clipping, or the clip threshold is measured
            # against scaled (huge) gradients and clips almost everything.
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        avg_loss = total_loss / len(self.train_loader.dataset)
        accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))
        return avg_loss, accuracy

    def _validate(self) -> Tuple[float, float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for beats, labels in self.val_loader:
                beats = beats.to(self.device)
                labels = labels.to(self.device)

                output = self.model(beats)
                logits = output[0] if isinstance(output, tuple) else output

                loss = self.criterion(logits, labels)
                total_loss += loss.item() * len(labels)

                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        avg_loss = total_loss / len(self.val_loader.dataset)
        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)

        metrics = calculate_all_metrics(
            all_labels_np, all_preds_np, num_classes=self.num_classes
        )
        return avg_loss, metrics["accuracy"], metrics

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "history": self.history,
            "model_name": self.model_name,
        }
        torch.save(checkpoint, self.best_checkpoint_path)
        logger.info(
            "  Checkpoint saved -> %s  (val_loss=%.6f)",
            self.best_checkpoint_path, val_loss,
        )

    # ------------------------------------------------------------------

    def train(self) -> Dict[str, List[float]]:
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("=" * 65)
        logger.info(
            "Training: %-25s  params=%s  device=%s  amp=%s",
            self.model_name.upper(), f"{n_params:,}", self.device, self.use_amp,
        )
        logger.info("=" * 65)

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.perf_counter()

            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc, val_metrics = self._validate()

            elapsed = time.perf_counter() - t0

            logger.info(
                "Epoch [%3d/%3d]  train_loss=%.4f  train_acc=%.4f  "
                "val_loss=%.4f  val_acc=%.4f  val_f1=%.4f  lr=%.2e  (%.1fs)",
                epoch, self.num_epochs, train_loss, train_acc,
                val_loss, val_acc, val_metrics["f1"],
                self.optimizer.param_groups[0]["lr"], elapsed,
            )

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_recall"].append(val_metrics["recall"])

            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self._save_checkpoint(epoch, val_loss)
                logger.info("  New best Macro-F1 = %.4f", self.best_val_f1)

            self.scheduler.step(val_metrics["f1"])

            if self.early_stopping(val_metrics["f1"]):
                logger.info("Early stopping triggered after epoch %d.", epoch)
                break

        plot_loss_curve(
            self.history["train_loss"], self.history["val_loss"],
            model_name=self.model_name, save_dir=self.plots_dir,
        )
        plot_accuracy_curve(
            self.history["train_acc"], self.history["val_acc"],
            model_name=self.model_name, save_dir=self.plots_dir,
        )

        logger.info("=" * 65)
        logger.info("Training complete. Best Macro-F1 = %.4f", self.best_val_f1)
        logger.info("=" * 65)
        return self.history

    def load_best_checkpoint(self) -> None:
        if not os.path.exists(self.best_checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.best_checkpoint_path}")
        checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            "Loaded best checkpoint from epoch %d  (val_loss=%.6f)",
            checkpoint["epoch"], checkpoint["val_loss"],
        )