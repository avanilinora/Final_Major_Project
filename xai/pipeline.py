"""
xai/pipeline.py
================
The core "explain one ECG recording, for all 12 leads, and save every
figure" routine — extracted so it can be called either:
  (a) once per class, over 5 representative test-set examples
      (generate_xai_report_assets.py), or
  (b) once, for a single new recording a user provides
      (explain_user_recording.py)

Both callers get identical folder structure and identical figures; only
how the input recording and its label/folder-name are chosen differs.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from explain_recording import explain_single_recording
from xai.report_figures import (
    plot_12lead_grid,
    plot_comparison_to_normal,
    plot_gradcam_single_lead,
    plot_lead_attention_bar,
    plot_technical_annotation,
    plot_temporal_attention_single_lead,
)


def lead_folder_name(lead_idx: int, lead_name: str) -> str:
    return f"lead_{lead_idx + 1:02d}_{lead_name}"


def generate_assets_for_recording(
    model: torch.nn.Module,
    x_np: np.ndarray,
    label: Optional[str],
    save_dir: str,
    class_names: list,
    lead_names: list,
    fs: int,
    device: torch.device,
    normal_signal_full: Optional[np.ndarray] = None,
    normal_class_name: Optional[str] = None,
) -> dict:
    """
    Run the full explanation pipeline for ONE recording and save every
    figure under `save_dir`, using the same folder layout for every
    caller:

        save_dir/
          lead_attention.png
          twelve_lead_grid.png
          lead_01_I/
            technical.png
            gradcam_saliency.png
            temporal_attention.png
            vs_normal.png          <- only if normal_signal_full is given
                                       AND label != normal_class_name

    Parameters
    ----------
    x_np : np.ndarray, shape (T, num_leads)
        A single, already-preprocessed recording (same preprocessing as
        training: band-pass filtered, wavelet-denoised, per-lead
        z-scored, fixed-length).
    label : str, optional
        Used for figure titles/captions. Pass the TRUE class name when
        it's known in advance (e.g. a test-set example). Pass None for a
        genuinely new, unlabelled recording — figure titles will then use
        the model's own predicted class name instead, since there is no
        other label to show.
    normal_signal_full : np.ndarray, optional
        A reference "normal" recording for the comparison-to-normal
        figures. If not provided, those figures are skipped entirely
        (this is a legitimate, expected mode — e.g. a deployed app might
        not ship the full test set, just the model).
    normal_class_name : str, optional
        If `label` (or the predicted class, when label is None) equals
        this, comparison-to-normal figures are skipped.

    Returns
    -------
    dict
        Same structure as one entry of xai_report_summary.json's
        "classes" dict — directly consumable by build_xai_report.py.
    """
    os.makedirs(save_dir, exist_ok=True)
    x = torch.from_numpy(x_np).float().unsqueeze(0)

    result = explain_single_recording(model, x, device)
    predicted_name = class_names[result["predicted_class"]]
    confidence = result["confidence"]

    # No pre-known label (a genuinely new recording) -> use the model's
    # own prediction for figure titles, since that's the only class name
    # available at plotting time.
    effective_label = label if label is not None else predicted_name

    lead_attn_path = os.path.join(save_dir, "lead_attention.png")
    plot_lead_attention_bar(result["lead_weights"], effective_label, lead_names, lead_attn_path)

    grid_path = os.path.join(save_dir, "twelve_lead_grid.png")
    plot_12lead_grid(x_np, fs, effective_label, grid_path, lead_names)

    summary = {
        "predicted_class": predicted_name,
        "confidence": confidence,
        "probs": {class_names[i]: float(p) for i, p in enumerate(result["probs"])},
        "lead_weights": {lead_names[i]: float(w) for i, w in enumerate(result["lead_weights"])},
        "lead_attention_figure": lead_attn_path,
        "twelve_lead_grid_figure": grid_path,
        "leads": {},
    }

    include_comparison = normal_signal_full is not None and effective_label != normal_class_name

    for lead_idx, lead_name in enumerate(lead_names):
        lead_dir = os.path.join(save_dir, lead_folder_name(lead_idx, lead_name))
        os.makedirs(lead_dir, exist_ok=True)

        signal = x_np[:, lead_idx]

        tech_result = plot_technical_annotation(
            signal, fs, effective_label, os.path.join(lead_dir, "technical.png"), lead_name=lead_name,
        )

        plot_gradcam_single_lead(
            signal, result["per_lead_cam"][lead_idx], lead_name, effective_label,
            os.path.join(lead_dir, "gradcam_saliency.png"),
            predicted_class=predicted_name, confidence=confidence,
        )

        plot_temporal_attention_single_lead(
            signal, result["temporal_attention"], lead_name, effective_label,
            os.path.join(lead_dir, "temporal_attention.png"),
        )

        lead_summary = {"technical": tech_result, "folder": lead_dir}

        if include_comparison:
            normal_signal = normal_signal_full[:, lead_idx]
            comp_result = plot_comparison_to_normal(
                signal, normal_signal, fs, effective_label,
                os.path.join(lead_dir, "vs_normal.png"), lead_name=lead_name,
            )
            lead_summary["comparison_to_normal"] = comp_result

        summary["leads"][lead_name] = lead_summary

    return summary