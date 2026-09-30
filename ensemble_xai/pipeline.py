"""
ensemble_xai/pipeline.py
==========================
generate_ensemble_assets_for_recording(): the ensemble equivalent of
xai/pipeline.py's generate_assets_for_recording(). Same folder layout
convention as the CNN-BiLSTM report, so both report types sit naturally
side by side:

    save_dir/
      lead_importance.png            <- ensemble, one per recording
      model_agreement.png            <- one per recording
      per_model_lead_importance.png  <- one per recording
      lead_time_heatmap.png          <- ensemble, one per recording
      lead_01_I/
        saliency_overlay.png
      lead_02_II/
        saliency_overlay.png
      ...
      lead_12_V6/
        saliency_overlay.png
"""

from __future__ import annotations

import os

import numpy as np

from ensemble_xai.occlusion import compute_occlusion_explanation
from ensemble_xai.report_figures import (
    plot_ensemble_lead_importance,
    plot_ensemble_lead_time_heatmap,
    plot_ensemble_saliency_overlay,
    plot_model_agreement_panel,
    plot_per_model_lead_importance_grid,
)


def lead_folder_name(lead_idx: int, lead_name: str) -> str:
    return f"lead_{lead_idx + 1:02d}_{lead_name}"


def generate_ensemble_assets_for_recording(
    x_np: np.ndarray,
    adapters: list,
    weights: np.ndarray,
    label: str,
    save_dir: str,
    class_names: list,
    lead_names: list,
    window_size: int = 250,
    stride: int = 250,
    combiner=None,
) -> dict:
    """
    x_np : (T, num_leads) — one preprocessed recording.
    adapters : list of ModelAdapter (see ensemble_xai/adapters.py).
    weights : ensemble weights, same order as adapters (ignored when combiner is given).
    combiner : optional StackedCombiner from ensemble_combiner.py -- pass it to explain
        the final stack+bias ensemble. The 'w=' labels then show how much the
        stacker leans on each model.
    label : used in figure titles. Pass the true class name when known,
        or None to use the ensemble's own predicted class (matches the
        convention used for the CNN-BiLSTM single-recording report).

    Returns a summary dict, directly JSON-serialisable and directly
    consumable by a report-builder (same shape convention as
    xai_report_summary.json).
    """
    os.makedirs(save_dir, exist_ok=True)

    result = compute_occlusion_explanation(
        x_np, adapters, weights, class_names, window_size=window_size, stride=stride,
        combiner=combiner,
    )
    predicted_name = class_names[result.predicted_class]
    effective_label = label if label is not None else predicted_name

    if combiner is not None:
        weights_dict = combiner.model_shares()
    else:
        weights_norm = weights / weights.sum()
        weights_dict = {a.name: float(w) for a, w in zip(adapters, weights_norm)}

    lead_importance_path = os.path.join(save_dir, "lead_importance.png")
    plot_ensemble_lead_importance(result.ensemble_per_lead_importance, lead_names, effective_label, lead_importance_path)

    heatmap_path = os.path.join(save_dir, "lead_time_heatmap.png")
    plot_ensemble_lead_time_heatmap(result.ensemble_lead_time_map, lead_names, effective_label, heatmap_path)

    agreement_path = os.path.join(save_dir, "model_agreement.png")
    plot_model_agreement_panel(
        result.per_model_pred, result.per_model_conf, result.per_model_agrees,
        predicted_name, result.confidence, weights_dict, effective_label, agreement_path,
    )

    per_model_grid_path = os.path.join(save_dir, "per_model_lead_importance.png")
    plot_per_model_lead_importance_grid(
        result.per_model_per_lead_importance, lead_names, effective_label, per_model_grid_path,
    )

    summary = {
        "predicted_class": predicted_name,
        "confidence": result.confidence,
        "probs": {class_names[i]: float(p) for i, p in enumerate(result.ensemble_probs)},
        "ensemble_weights": weights_dict,
        "per_model_prediction": {n: class_names[p] for n, p in result.per_model_pred.items()},
        "per_model_confidence": result.per_model_conf,
        "per_model_agrees_with_ensemble": result.per_model_agrees,
        "n_models_agreeing": sum(result.per_model_agrees.values()),
        "n_models_total": len(adapters),
        "ensemble_lead_importance": {lead_names[i]: float(v) for i, v in enumerate(result.ensemble_per_lead_importance)},
        "lead_importance_figure": lead_importance_path,
        "lead_time_heatmap_figure": heatmap_path,
        "model_agreement_figure": agreement_path,
        "per_model_lead_importance_figure": per_model_grid_path,
        "leads": {},
    }

    for lead_idx, lead_name in enumerate(lead_names):
        lead_dir = os.path.join(save_dir, lead_folder_name(lead_idx, lead_name))
        os.makedirs(lead_dir, exist_ok=True)
        signal = x_np[:, lead_idx]
        saliency_path = os.path.join(lead_dir, "saliency_overlay.png")
        plot_ensemble_saliency_overlay(
            signal, result.ensemble_lead_time_map[lead_idx], lead_name, effective_label,
            predicted_name, result.confidence, saliency_path,
        )
        summary["leads"][lead_name] = {
            "folder": lead_dir,
            "saliency_figure": saliency_path,
            "importance_weight": float(result.ensemble_per_lead_importance[lead_idx]),
        }

    return summary