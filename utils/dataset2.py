"""
utils/dataset2.py
==================
PyTorch Dataset and DataLoader factory for preprocessed ECG arrays.

Works for BOTH tracks in this project:
  * MIT-BIH beat-level samples, shape (N, 360, 2)
  * PhysioNet record-level samples, shape (N, 5000, 12)



Usage
-----
    from utils.dataset2 import ECGDataset, build_dataloaders

    # PhysioNet
    from config_physionet import PhysioNetConfig as PNConfig
    PNConfig.load_label_vocab()
    train_loader, val_loader, test_loader = build_dataloaders(
        processed_path=PNConfig.PROCESSED_PATH,
        batch_size=PNConfig.BATCH_SIZE,
        num_classes=PNConfig.NUM_CLASSES,
    )
"""

from __future__ import annotations
import random
import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config_physionet import PhysioNetConfig as Config

logger = logging.getLogger(__name__)


# ===========================================================================
# Dataset
# ===========================================================================

class ECGDataset(Dataset):
    """
    PyTorch Dataset for preprocessed ECG arrays (beat- or record-level).

    Parameters
    ----------
    X : np.ndarray, shape (N, seq_len, num_leads)
        Samples produced by either preprocessing pipeline.
    y : np.ndarray, shape (N,)
        Integer class labels.
    num_classes : int, optional
        Total number of classes in this task. Defaults to
        ``Config.NUM_CLASSES`` (PhysioNet, per the fixed import above) —
        pass this explicitly if  a different class count is needed
    transform : callable, optional
        Optional transform applied to each sample tensor before returning.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_classes: Optional[int] = None,
        transform: Optional[callable] = None,
        augment=False,
    ) -> None:
        super().__init__()

        if X.ndim != 3:
            raise ValueError(
                f"Expected X with shape (N, seq_len, num_leads), got {X.shape}."
            )

        if len(X) != len(y):
            raise ValueError(
                f"X and y length mismatch: {len(X)} vs {len(y)}."
            )

        self.augment = augment

        self.num_classes = num_classes if num_classes is not None else Config.NUM_CLASSES

        self.X: Tensor = torch.from_numpy(X.astype(np.float32))
        self.y: Tensor = torch.from_numpy(y.astype(np.int64))

        self.transform = transform

        logger.debug(
            "ECGDataset created: N=%d  seq_len=%d  num_leads=%d  classes=%s",
            len(self.y), self.X.shape[1], self.X.shape[2],
            {int(c): int((self.y == c).sum()) for c in self.y.unique()},
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.y)

    def augment_signal(self, signal):

        # Gaussian noise
        if random.random() < 0.5:
            noise = torch.randn_like(signal) * 0.01
            signal = signal + noise

        # Random amplitude scaling
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            signal = signal * scale

        # Small temporal shift
        if random.random() < 0.5:
            shift = random.randint(-20, 20)
            signal = torch.roll(signal, shifts=shift, dims=0)

        return signal

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        sample = self.X[idx].clone()
        if self.augment:
            sample = self.augment_signal(sample)
        label: Tensor = self.y[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    # ------------------------------------------------------------------
    # Utility properties
    # ------------------------------------------------------------------

    @property
    def num_classes_present(self) -> int:
        """Number of unique classes actually present in this split."""
        return int(self.y.unique().numel())

    @property
    def class_counts(self) -> np.ndarray:
        """Array of per-class sample counts (length = ``self.num_classes``)."""
        counts = np.zeros(self.num_classes, dtype=np.int64)
        y_np = self.y.numpy()
        for i in range(self.num_classes):
            counts[i] = int((y_np == i).sum())
        return counts

    @property
    def class_weights(self) -> Tensor:
        """
        Inverse-sqrt-frequency class weights for
        ``torch.nn.CrossEntropyLoss(weight=...)``, normalised to sum to
        ``self.num_classes``.
        """
        counts = self.class_counts.astype(np.float32)
        counts = np.where(counts == 0, 1.0, counts)
        weights = 1.0 / np.sqrt(counts)
        weights = weights / weights.sum() * self.num_classes
        return torch.from_numpy(weights.astype(np.float32))


# ===========================================================================
# Weighted sampler
# ===========================================================================

def build_weighted_sampler(dataset: ECGDataset) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that up-samples minority classes."""
    y_np: np.ndarray = dataset.y.numpy()
    class_counts = dataset.class_counts.astype(np.float32)
    class_weights = np.where(class_counts == 0, 0.0, 1.0 / class_counts)
    sample_weights = class_weights[y_np]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights.astype(np.float32)),
        num_samples=len(dataset),
        replacement=True,
    )
    return sampler


# ===========================================================================
# DataLoader factory
# ===========================================================================

def build_dataloaders(
    processed_path: str = Config.PROCESSED_PATH,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    pin_memory: bool = True,
    num_classes: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load preprocessed NumPy arrays and return ready-to-use DataLoaders.

    Parameters
    ----------
    processed_path : str
        Directory containing ``X_train.npy``, ``y_train.npy``, etc.
        Pass ``PhysioNetConfig.PROCESSED_PATH`` for the PhysioNet track.
    num_classes : int, optional
        Forwarded to ``ECGDataset``. Defaults to ``Config.NUM_CLASSES``
        (PhysioNet, per the fixed import above).

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    def _load(name: str) -> np.ndarray:
        path = os.path.join(processed_path, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Processed file not found: {path}\n"
                "Run the matching preprocessing script first."
            )
        return np.load(path)

    X_train = _load("X_train")
    y_train = _load("y_train")
    X_val = _load("X_val")
    y_val = _load("y_val")
    X_test = _load("X_test")
    y_test = _load("y_test")

    train_ds = ECGDataset(X_train, y_train, num_classes=num_classes, augment=True)
    val_ds = ECGDataset(X_val, y_val, num_classes=num_classes)
    test_ds = ECGDataset(X_test, y_test, num_classes=num_classes)

    logger.info(
        "Datasets — train: %d | val: %d | test: %d",
        len(train_ds), len(val_ds), len(test_ds),
    )
    logger.info("Train class counts: %s", train_ds.class_counts.tolist())

    _pin = pin_memory and torch.cuda.is_available()

    if use_weighted_sampler:
        sampler = build_weighted_sampler(train_ds)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=_pin, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=_pin, drop_last=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=_pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=_pin,
    )

    return train_loader, val_loader, test_loader
