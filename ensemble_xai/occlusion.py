"""
ensemble_xai/occlusion.py
===========================
Model-agnostic explainability for the ensemble, via occlusion sensitivity
(Zeiler & Fergus, 2014 — the standard black-box alternative to gradient
methods, used here because the ensemble mixes gradient-based, tree-based,
and unknown-architecture models with no shared internal structure to hook
into).

Method: zero out one (lead, time-window) region at a time, re-run every
adapter, and measure how much the ENSEMBLE's confidence in its own
original predicted class drops. A large drop means that region was load-
bearing evidence for the decision; no drop means the model didn't need it.

This produces the same three outputs the CNN-BiLSTM Grad-CAM++ pipeline
does — per-lead importance, a per-lead heatmap, an aggregated time trace
— plus one thing that pipeline couldn't: the SAME analysis re-run per
individual model, so agreement/disagreement in *what each model looked
at* is visible, not just agreement in the final class vote.

*** Honesty note carried into every output of this module ***
Occlusion importance is an approximation: it measures the marginal effect
of removing one region while others stay intact, and can under-credit
regions whose signal is redundant with nearby leads/time. This is
reported as a limitation everywhere this module's output is used, the
same way the P/QRS/T window approximation was flagged in the CNN-BiLSTM
report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class OcclusionResult:
    predicted_class: int
    confidence: float
    ensemble_probs: np.ndarray                    # (C,)
    per_model_pred: Dict[str, int]
    per_model_conf: Dict[str, float]
    per_model_agrees: Dict[str, bool]
    ensemble_per_lead_importance: np.ndarray       # (L,) normalised to sum=1
    ensemble_lead_time_map: np.ndarray             # (L, T) upsampled to full length
    ensemble_aggregated_trace: np.ndarray          # (T,) normalised 0-1
    per_model_per_lead_importance: Dict[str, np.ndarray] = field(default_factory=dict)


def _occlude_batch(x: np.ndarray, lead_idx: int, window_starts: np.ndarray, window_size: int) -> np.ndarray:
    """x: (T, L). Returns (n_windows, T, L), each copy with one window on
    one lead zeroed (0 = the dataset's own mean after z-score normalisation,
    so this is a neutral baseline, not an arbitrary distortion)."""
    T = x.shape[0]
    batch = np.repeat(x[None, :, :], len(window_starts), axis=0)
    for i, start in enumerate(window_starts):
        end = min(start + window_size, T)
        batch[i, start:end, lead_idx] = 0.0
    return batch


def _weighted_ensemble(probs_per_model: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    weights = weights / weights.sum()
    out = np.zeros_like(probs_per_model[0])
    for w, p in zip(weights, probs_per_model):
        out += w * p
    return out


def _combine(probs_per_model: List[np.ndarray], weights: np.ndarray, combiner=None) -> np.ndarray:
    """Final ensemble probabilities. With a combiner (the saved stack+bias model from
    ensemble_combiner.py) use it; otherwise fall back to the plain weighted average."""
    if combiner is not None:
        return combiner.predict_proba(probs_per_model)
    return _weighted_ensemble(probs_per_model, weights)


def compute_occlusion_explanation(
    x: np.ndarray,
    adapters: list,
    weights: np.ndarray,
    class_names: list,
    window_size: int = 250,
    stride: int = 250,
    batch_size: int = 64,
    combiner=None,
) -> OcclusionResult:
    """
    x : (T, L) — ONE preprocessed recording.
    adapters : list of ModelAdapter, each exposing .name and .predict_proba.
    weights : (len(adapters),) — the same optimized ensemble weights used
        for reporting, so the explanation matches the numbers you publish.
        Ignored when `combiner` is given.
    combiner : optional StackedCombiner (ensemble_combiner.py). Pass it so the
        explanation is of the stack+bias ensemble you actually report.
    """
    T, L = x.shape
    num_classes = len(class_names)

    # --- Baseline: original, un-occluded prediction ---
    orig_probs_per_model = [a.predict_proba(x[None, :, :])[0] for a in adapters]
    orig_ensemble = _combine([p[None, :] for p in orig_probs_per_model], weights, combiner)[0]
    predicted_class = int(orig_ensemble.argmax())
    confidence = float(orig_ensemble[predicted_class])

    per_model_pred = {a.name: int(p.argmax()) for a, p in zip(adapters, orig_probs_per_model)}
    per_model_conf = {a.name: float(p.max()) for a, p in zip(adapters, orig_probs_per_model)}
    per_model_agrees = {name: (pred == predicted_class) for name, pred in per_model_pred.items()}

    # --- Occlusion sweep: every (lead, window) pair, batched per lead ---
    window_starts = np.arange(0, T, stride)
    n_windows = len(window_starts)

    ensemble_drop = np.zeros((L, n_windows))           # importance for the ensemble
    per_model_drop = {a.name: np.zeros((L, n_windows)) for a in adapters}

    for lead_idx in range(L):
        occluded = _occlude_batch(x, lead_idx, window_starts, window_size)  # (n_windows, T, L)

        probs_per_model_occ = []
        for a in adapters:
            outs = []
            for i in range(0, n_windows, batch_size):
                outs.append(a.predict_proba(occluded[i:i + batch_size]))
            probs_per_model_occ.append(np.vstack(outs))  # (n_windows, C)

        ensemble_occ = _combine(probs_per_model_occ, weights, combiner)   # (n_windows, C)

        # Drop in confidence for the ORIGINAL predicted class -> importance
        ensemble_drop[lead_idx, :] = confidence - ensemble_occ[:, predicted_class]
        for a, p_occ in zip(adapters, probs_per_model_occ):
            per_model_drop[a.name][lead_idx, :] = per_model_conf[a.name] - p_occ[:, per_model_pred[a.name]]

    # Negative "importance" (occlusion INCREASED confidence) is clipped to
    # zero — that region actively argued against the prediction, which is
    # a different (and rarer) thing than "supported" it; kept at zero here
    # to keep the saliency map's semantics simple: brighter = more
    # supporting evidence. This clipping is applied consistently, not
    # selectively, so it doesn't hide anything.
    ensemble_drop = np.clip(ensemble_drop, 0, None)

    # --- Per-lead importance (normalised to sum to 1, same convention as the lead-attention bar chart) ---
    lead_importance = ensemble_drop.sum(axis=1)
    if lead_importance.sum() > 0:
        lead_importance = lead_importance / lead_importance.sum()
    else:
        lead_importance = np.full(L, 1.0 / L)

    # --- Upsample window-level map to full signal length ---
    lead_time_map = np.zeros((L, T))
    x_old = window_starts + window_size / 2.0
    x_new = np.arange(T)
    for lead_idx in range(L):
        row = ensemble_drop[lead_idx]
        if row.max() > 0:
            row = row / row.max()
        lead_time_map[lead_idx] = np.interp(x_new, x_old, row)

    # --- Aggregated trace, weighted by lead importance (same convention as Grad-CAM++ aggregation) ---
    aggregated = (lead_time_map * lead_importance[:, None]).sum(axis=0)
    if aggregated.max() > 0:
        aggregated = (aggregated - aggregated.min()) / (aggregated.max() - aggregated.min() + 1e-8)

    per_model_lead_importance = {}
    for name, drop in per_model_drop.items():
        drop = np.clip(drop, 0, None)
        s = drop.sum(axis=1)
        per_model_lead_importance[name] = s / s.sum() if s.sum() > 0 else np.full(L, 1.0 / L)

    return OcclusionResult(
        predicted_class=predicted_class,
        confidence=confidence,
        ensemble_probs=orig_ensemble,
        per_model_pred=per_model_pred,
        per_model_conf=per_model_conf,
        per_model_agrees=per_model_agrees,
        ensemble_per_lead_importance=lead_importance,
        ensemble_lead_time_map=lead_time_map,
        ensemble_aggregated_trace=aggregated,
        per_model_per_lead_importance=per_model_lead_importance,
    )