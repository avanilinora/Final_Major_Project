from __future__ import annotations
import logging, os
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
_PALETTE = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED"]


def plot_loss_curve(train_losses, val_losses, model_name="model", save_dir="outputs/plots", show=False):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, color=_PALETTE[0], label="Train Loss", marker="o", markersize=3)
    ax.plot(epochs, val_losses, color=_PALETTE[2], label="Val Loss", marker="s", markersize=3, linestyle="--")
    best_epoch = int(np.argmin(val_losses)) + 1
    ax.axvline(x=best_epoch, color="grey", linestyle=":", label=f"Best val (epoch {best_epoch})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(f"Loss Curve — {model_name.upper()}")
    ax.legend(); fig.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_loss_curve.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def plot_accuracy_curve(train_accs, val_accs, model_name="model", save_dir="outputs/plots", show=False):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_accs) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [a * 100 for a in train_accs], color=_PALETTE[0], label="Train Acc", marker="o", markersize=3)
    ax.plot(epochs, [a * 100 for a in val_accs], color=_PALETTE[1], label="Val Acc", marker="s", markersize=3, linestyle="--")
    best_epoch = int(np.argmax(val_accs)) + 1
    ax.axvline(x=best_epoch, color="grey", linestyle=":", label=f"Best val (epoch {best_epoch})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); ax.set_title(f"Accuracy Curve — {model_name.upper()}")
    ax.legend(); fig.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_accuracy_curve.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def plot_confusion_matrix(y_true, y_pred, model_name="model", class_names=None, normalise=True,
                           save_dir="outputs/confusion_matrices", show=False):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix as sk_cm
    os.makedirs(save_dir, exist_ok=True)
    if class_names is None:
        raise ValueError("plot_confusion_matrix requires class_names.")
    num_classes = len(class_names)
    cm = sk_cm(y_true, y_pred, labels=list(range(num_classes)))
    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_plot = cm.astype(float) / row_sums
        fmt = ".2f"
    else:
        cm_plot = cm; fmt = "d"
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name.upper()}")
    fig.tight_layout()
    tag = "normalised" if normalise else "raw"
    path = os.path.join(save_dir, f"{model_name}_confusion_matrix_{tag}.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def plot_roc_curve(y_true, y_prob, model_name="model", class_names=None, save_dir="outputs/plots", show=False):
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import label_binarize
    os.makedirs(save_dir, exist_ok=True)
    if class_names is None:
        raise ValueError("plot_roc_curve requires class_names.")
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls_name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=_PALETTE[i % len(_PALETTE)], label=f"Class {cls_name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="lightgrey", linestyle=":")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {model_name.upper()}")
    ax.legend(loc="lower right", fontsize=9); fig.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_roc_curve.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def plot_reliability_diagram(confidences: np.ndarray, correct: np.ndarray, model_name="model",
                              n_bins=10, save_dir="outputs/plots", show=False, title_suffix=""):
    """
    Reliability diagram: predicted confidence vs. observed accuracy.
    Returns (save_path, ece).
    """
    os.makedirs(save_dir, exist_ok=True)
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    bin_idx = np.clip(np.digitize(confidences, bin_edges[1:-1]), 0, n_bins - 1)

    for b in range(n_bins):
        mask = bin_idx == b
        bin_counts[b] = mask.sum()
        if bin_counts[b] > 0:
            bin_accs[b] = correct[mask].mean()
            bin_confs[b] = confidences[mask].mean()

    n_total = len(confidences)
    ece = float(np.sum(bin_counts / n_total * np.abs(bin_accs - bin_confs))) if n_total > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Perfect calibration")
    widths = np.diff(bin_edges)
    ax.bar(bin_edges[:-1], bin_accs, width=widths, align="edge", color=_PALETTE[0],
           alpha=0.75, edgecolor="black", linewidth=0.5, label="Observed accuracy")
    for b in range(n_bins):
        if bin_counts[b] > 0:
            ax.text((bin_edges[b] + bin_edges[b+1]) / 2, bin_accs[b] + 0.02, f"{int(bin_counts[b])}",
                    ha="center", fontsize=7, color="dimgrey")
    ax.set_xlabel("Predicted confidence"); ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_title(f"Reliability Diagram — {model_name.upper()}{title_suffix}\nECE = {ece:.4f}")
    ax.legend(loc="upper left"); fig.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_reliability_diagram{title_suffix.replace(' ', '_')}.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path, ece