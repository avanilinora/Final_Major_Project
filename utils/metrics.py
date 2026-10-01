"""
utils/metrics.py
=================
Classification metric functions used during training, validation, and
evaluation.



Functions
---------
calculate_accuracy()
calculate_precision()
calculate_recall()
calculate_f1()
calculate_specificity()
calculate_all_metrics()
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Type alias accepted by all public functions
ArrayLike = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert a PyTorch Tensor or NumPy array to a 1-D NumPy int64 array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.int64)
    return np.asarray(x, dtype=np.int64)


def _resolve_labels(y_true_np: np.ndarray, y_pred_np: np.ndarray,
                     num_classes: Optional[int]) -> list[int]:
    """
    Return the explicit label list to pass to sklearn.

    If `num_classes` is given, use range(num_classes) so every class is
    represented even if absent from this particular batch/split (this is
    what is needed for evaluation reports and confusion matrices). If not
    given, fall back to whatever labels are actually present in the data
    — safer than silently assuming a fixed class count belonging to a
    different track.
    """
    if num_classes is not None:
        return list(range(num_classes))
    return sorted(set(y_true_np.tolist()) | set(y_pred_np.tolist()))


# ===========================================================================
# Public metric functions
# ===========================================================================

def calculate_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Overall (micro) classification accuracy."""
    return float(accuracy_score(_to_numpy(y_true), _to_numpy(y_pred)))


def calculate_precision(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "macro",
    zero_division: int = 0,
    num_classes: Optional[int] = None,
) -> float:
    """Precision (default: macro-averaged across classes)."""
    y_true_np, y_pred_np = _to_numpy(y_true), _to_numpy(y_pred)
    return float(
        precision_score(
            y_true_np, y_pred_np,
            average=average,
            zero_division=zero_division,
            labels=_resolve_labels(y_true_np, y_pred_np, num_classes),
        )
    )


def calculate_recall(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "macro",
    zero_division: int = 0,
    num_classes: Optional[int] = None,
) -> float:
    """Recall / sensitivity (default: macro-averaged)."""
    y_true_np, y_pred_np = _to_numpy(y_true), _to_numpy(y_pred)
    return float(
        recall_score(
            y_true_np, y_pred_np,
            average=average,
            zero_division=zero_division,
            labels=_resolve_labels(y_true_np, y_pred_np, num_classes),
        )
    )


def calculate_f1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "macro",
    zero_division: int = 0,
    num_classes: Optional[int] = None,
) -> float:
    """F1 score (default: macro-averaged)."""
    y_true_np, y_pred_np = _to_numpy(y_true), _to_numpy(y_pred)
    return float(
        f1_score(
            y_true_np, y_pred_np,
            average=average,
            zero_division=zero_division,
            labels=_resolve_labels(y_true_np, y_pred_np, num_classes),
        )
    )


def calculate_specificity(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "macro",
    num_classes: Optional[int] = None,
) -> float:
    """
    Specificity (true negative rate), macro-averaged by default.
    Specificity for class i = TN_i / (TN_i + FP_i), one-vs-rest.
    """
    y_true_np, y_pred_np = _to_numpy(y_true), _to_numpy(y_pred)
    labels = _resolve_labels(y_true_np, y_pred_np, num_classes)
    n = len(labels)

    cm = confusion_matrix(y_true_np, y_pred_np, labels=labels)

    specificities: list[float] = []
    supports: list[int] = []

    for i in range(n):
        tn = int(cm.sum()) - int(cm[i, :].sum()) - int(cm[:, i].sum()) + int(cm[i, i])
        fp = int(cm[:, i].sum()) - int(cm[i, i])
        denom = tn + fp
        specificities.append(tn / denom if denom > 0 else 0.0)
        supports.append(int(cm[i, :].sum()))

    if average == "macro":
        return float(np.mean(specificities))
    elif average == "weighted":
        total = sum(supports)
        if total == 0:
            return 0.0
        return float(sum(s * w for s, w in zip(specificities, supports)) / total)
    elif average == "micro":
        total_tn, total_fp = 0, 0
        for i in range(n):
            tn = (int(cm.sum()) - int(cm[i, :].sum())
                  - int(cm[:, i].sum()) + int(cm[i, i]))
            fp = int(cm[:, i].sum()) - int(cm[i, i])
            total_tn += tn
            total_fp += fp
        denom = total_tn + total_fp
        return float(total_tn / denom) if denom > 0 else 0.0
    else:
        raise ValueError(
            f"Unknown average mode '{average}'. Choose from 'macro', 'weighted', 'micro'."
        )


# ===========================================================================
# Convenience aggregator
# ===========================================================================

def calculate_all_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "macro",
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute all standard metrics in a single call.

    num_classes : int, optional
        Pass this explicitly (e.g. PhysioNetConfig.NUM_CLASSES) whenever
        you're not on the Phase-1 4-class track — see module docstring.
    """
    return {
        "accuracy":    calculate_accuracy(y_true, y_pred),
        "precision":   calculate_precision(y_true, y_pred, average=average, num_classes=num_classes),
        "recall":      calculate_recall(y_true, y_pred, average=average, num_classes=num_classes),
        "f1":          calculate_f1(y_true, y_pred, average=average, num_classes=num_classes),
        "specificity": calculate_specificity(y_true, y_pred, average=average, num_classes=num_classes),
    }
