"""
ensemble_xai/report_figures.py
================================
Plotting for ensemble occlusion explanations. Deliberately mirrors the
visual language of xai/report_figures.py (the CNN-BiLSTM report) so a
reader flipping between the two report types isn't learning a new visual
convention each time.
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

_PALETTE = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED", "#0891B2"]


def plot_ensemble_lead_importance(per_lead_importance: np.ndarray, lead_names: list,
                                   label: str, save_path: str) -> str:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(lead_names, per_lead_importance, color="#2563EB")
    ax.set_ylabel("Occlusion importance (normalised)")
    ax.set_title(f"Ensemble lead importance — {label}\n(occlusion-based, model-agnostic)")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path


def plot_ensemble_saliency_overlay(signal: np.ndarray, aggregated_trace: np.ndarray,
                                    lead_name: str, label: str, predicted_class: str,
                                    confidence: float, save_path: str) -> str:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    T = len(signal)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(signal, color="black", linewidth=0.9)
    ax2 = ax.twinx()
    ax2.fill_between(np.arange(T), 0, aggregated_trace, color="red", alpha=0.3)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Occlusion saliency (ensemble)", color="red")
    ax.set_title(f"Lead {lead_name} — {label} — Ensemble occlusion saliency\n"
                 f"Predicted: {predicted_class} (confidence {confidence:.1%})")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path


def plot_ensemble_lead_time_heatmap(lead_time_map: np.ndarray, lead_names: list,
                                     label: str, save_path: str) -> str:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    T = lead_time_map.shape[1]
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(lead_time_map, aspect="auto", cmap="hot", extent=[0, T, len(lead_names), 0])
    ax.set_yticks(np.arange(len(lead_names)) + 0.5)
    ax.set_yticklabels(lead_names)
    ax.set_xlabel("Time (samples)")
    ax.set_title(f"Ensemble occlusion saliency, per lead — {label}")
    fig.colorbar(im, ax=ax, label="Saliency")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path


def plot_model_agreement_panel(per_model_pred: dict, per_model_conf: dict, per_model_agrees: dict,
                                ensemble_pred_name: str, ensemble_conf: float,
                                weights: dict, label: str, save_path: str) -> str:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    names = list(per_model_pred.keys())
    confs = [per_model_conf[n] for n in names]
    colors = ["#16A34A" if per_model_agrees[n] else "#DC2626" for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names + ["ENSEMBLE"], confs + [ensemble_conf], color=colors + ["#1F2937"])
    for i, n in enumerate(names):
        ax.text(i, confs[i] + 0.02, per_model_pred[n], ha="center", fontsize=9)
        ax.text(i, -0.06, f"w={weights.get(n, 0):.2f}", ha="center", fontsize=8, color="dimgrey")
    ax.text(len(names), ensemble_conf + 0.02, ensemble_pred_name, ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(-0.1, 1.15)
    ax.set_ylabel("Confidence in own predicted class")
    agree_count = sum(per_model_agrees.values())
    ax.set_title(f"{label}\nModel agreement: {agree_count}/{len(names)} models agree with the ensemble "
                 f"(green = agrees, red = disagrees)")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path


def plot_per_model_lead_importance_grid(per_model_importance: dict, lead_names: list,
                                         label: str, save_path: str) -> str:
    """Small multiples: each model's own occlusion-based lead importance,
    side by side — the direct visual cross-check for 'did the models look
    at similar evidence, or different evidence that happened to agree'."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    names = list(per_model_importance.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, name, color in zip(axes, names, _PALETTE):
        ax.bar(lead_names, per_model_importance[name], color=color)
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=60)
    fig.suptitle(f"Per-model lead importance (occlusion) — {label}")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path