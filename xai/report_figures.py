"""
xai/report_figures.py
======================
Figure generators for the XAI report: an annotated "how to read this ECG"
technical figure, a normal-vs-abnormal comparison figure, and a full
12-lead grid — built on top of xai/ecg_annotation.py's R-peak detection.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from xai.ecg_annotation import (
    approximate_pqrst_windows,
    detect_r_peaks,
    estimate_heart_rate,
    find_clean_beat_index,
)

_PALETTE = {"P": "#16A34A", "QRS": "#DC2626", "T": "#2563EB"}


def plot_technical_annotation(
    signal: np.ndarray, fs: int, class_name: str, save_path: str, lead_name: str = "II",
) -> dict:
    """
    Full-signal waveform with detected R-peaks marked, plus a zoomed inset
    of one beat showing approximate P/QRS/T windows. Returns a dict of the
    computed numbers (duration, heart rate, etc.) for reuse in report text.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    T = len(signal)
    duration_s = T / fs
    time_axis = np.arange(T) / fs

    r_peaks = detect_r_peaks(signal, fs)
    heart_rate = estimate_heart_rate(r_peaks, fs)
    beat_idx = find_clean_beat_index(r_peaks, T, fs)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1.4]})

    # --- Panel 1: full signal with R-peaks marked ---
    axes[0].plot(time_axis, signal, color="black", linewidth=0.9)
    if len(r_peaks) > 0:
        axes[0].scatter(r_peaks / fs, signal[r_peaks], color="#DC2626", s=35, zorder=5, label="Detected R-peak (one per heartbeat)")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude (normalised)")
    hr_text = f"{heart_rate:.0f} bpm" if heart_rate else "n/a"
    axes[0].set_title(
        f"Lead {lead_name} — {class_name}\n"
        f"{T} samples @ {fs} Hz = {duration_s:.1f}s recording   |   "
        f"{len(r_peaks)} heartbeats detected   |   estimated rate: {hr_text}"
    )
    axes[0].legend(loc="upper right")

    # --- Panel 2: zoomed single-beat inset with approximate P/QRS/T ---
    result = {"duration_s": duration_s, "num_samples": T, "fs": fs,
              "num_beats_detected": len(r_peaks), "heart_rate_bpm": heart_rate}

    if beat_idx is not None:
        r = r_peaks[beat_idx]
        prev_r = r_peaks[beat_idx - 1]
        next_r = r_peaks[beat_idx + 1]
        windows = approximate_pqrst_windows(r, prev_r, next_r, fs)

        zoom_start = max(0, prev_r + (r - prev_r) // 2)
        zoom_end = min(T, next_r - (next_r - r) // 2)
        zoom_time = np.arange(zoom_start, zoom_end) / fs
        zoom_signal = signal[zoom_start:zoom_end]

        axes[1].plot(zoom_time, zoom_signal, color="black", linewidth=1.3)
        # Extra headroom above the tallest point in this zoomed window, so
        # the P/QRS/T text labels (placed just above the signal) never
        # collide with the axes' own title — this was a real, visible bug:
        # on a tall QRS peak, the default y-limit put the "QRS" label
        # directly on top of the subplot title text.
        y_range = zoom_signal.max() - zoom_signal.min()
        axes[1].set_ylim(top=zoom_signal.max() + 0.45 * y_range)
        for label, (ws, we) in windows.items():
            ws_c, we_c = max(ws, zoom_start), min(we, zoom_end)
            if we_c > ws_c:
                axes[1].axvspan(ws_c / fs, we_c / fs, color=_PALETTE[label], alpha=0.18)
                mid = (ws_c + we_c) / 2 / fs
                y_top = zoom_signal.max()
                axes[1].text(mid, y_top * 1.08, label, ha="center", fontsize=11,
                              color=_PALETTE[label], fontweight="bold")
        axes[1].scatter([r / fs], [signal[r]], color="#DC2626", s=60, zorder=5)
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_ylabel("Amplitude (normalised)")
        axes[1].set_title(
            "One heartbeat, zoomed in — approximate P-wave / QRS-complex / T-wave regions\n"
            "(illustrative timing windows, not a diagnostic-grade wave-delineation output)"
        )
        result["pqrst_windows_sample_indices"] = {k: [int(v[0]), int(v[1])] for k, v in windows.items()}
    else:
        axes[1].text(0.5, 0.5, "Not enough detected beats to show a zoomed single-beat view.",
                      ha="center", va="center", transform=axes[1].transAxes)
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    result["save_path"] = save_path
    return result


def plot_comparison_to_normal(
    signal_abnormal: np.ndarray, signal_normal: np.ndarray, fs: int,
    class_name: str, save_path: str, lead_name: str = "II",
) -> dict:
    """
    Stacked comparison: the abnormal-class recording above, the Normal
    (Sinus Rhythm) recording below, same lead, same time axis, both with
    R-peaks marked so the rate/rhythm difference is visually direct.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    T = min(len(signal_abnormal), len(signal_normal))
    time_axis = np.arange(T) / fs

    peaks_abn = detect_r_peaks(signal_abnormal[:T], fs)
    peaks_norm = detect_r_peaks(signal_normal[:T], fs)
    hr_abn = estimate_heart_rate(peaks_abn, fs)
    hr_norm = estimate_heart_rate(peaks_norm, fs)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(time_axis, signal_abnormal[:T], color="#DC2626", linewidth=0.9)
    if len(peaks_abn) > 0:
        axes[0].scatter(peaks_abn / fs, signal_abnormal[:T][peaks_abn], color="black", s=25, zorder=5)
    hr_abn_text = f"{hr_abn:.0f} bpm" if hr_abn else "n/a"
    axes[0].set_title(f"{class_name}  —  {len(peaks_abn)} beats detected, estimated rate {hr_abn_text}")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(time_axis, signal_normal[:T], color="#16A34A", linewidth=0.9)
    if len(peaks_norm) > 0:
        axes[1].scatter(peaks_norm / fs, signal_normal[:T][peaks_norm], color="black", s=25, zorder=5)
    hr_norm_text = f"{hr_norm:.0f} bpm" if hr_norm else "n/a"
    axes[1].set_title(f"Normal reference (Sinus Rhythm)  —  {len(peaks_norm)} beats detected, estimated rate {hr_norm_text}")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude")

    fig.suptitle(f"Lead {lead_name} — {class_name} vs. Normal Sinus Rhythm", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return {
        "class_heart_rate_bpm": hr_abn, "normal_heart_rate_bpm": hr_norm,
        "class_beats_detected": len(peaks_abn), "normal_beats_detected": len(peaks_norm),
        "save_path": save_path,
    }


def plot_lead_attention_bar(
    lead_weights: np.ndarray, class_name: str, lead_names: list, save_path: str,
) -> str:
    """
    Standalone lead-attention bar chart — ONE per class (not per lead),
    since this is a single set of 12 weights summarising how much each
    lead contributed, not a value that differs "per lead folder."
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(lead_names, lead_weights, color="#2563EB")
    ax.set_ylabel("Attention weight")
    ax.set_title(f"Lead-attention weights — {class_name}\n(which of the 12 leads the model relied on most)")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path


def plot_gradcam_single_lead(
    signal: np.ndarray, cam_trace: np.ndarray, lead_name: str, class_name: str, save_path: str,
    predicted_class: Optional[str] = None, confidence: Optional[float] = None,
) -> str:
    """
    ONE lead's own raw waveform with THAT LEAD'S OWN Grad-CAM++ saliency
    trace overlaid (per_lead_cam[lead_idx], not the cross-lead aggregated
    trace) — this is a genuinely per-lead explanation, unlike temporal
    attention below.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    T = len(signal)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(signal, color="black", linewidth=0.9)
    ax2 = ax.twinx()
    ax2.fill_between(np.arange(T), 0, cam_trace, color="red", alpha=0.3)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Grad-CAM++ saliency (this lead)", color="red")
    title = f"Lead {lead_name} — {class_name} — Grad-CAM++ saliency"
    if predicted_class is not None:
        title += f"\nPredicted: {predicted_class}" + (f"  (confidence {confidence:.1%})" if confidence is not None else "")
    ax.set_title(title)
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path


def plot_temporal_attention_single_lead(
    signal: np.ndarray, temporal_attention: np.ndarray, lead_name: str, class_name: str, save_path: str,
) -> str:
    """
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    T = len(signal)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(signal, color="black", linewidth=0.9)
    ax2 = ax.twinx()
    ax2.fill_between(np.arange(T), 0, temporal_attention, color="blue", alpha=0.3)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Temporal attention (shared across leads)", color="blue")
    ax.set_title(
        f"Lead {lead_name} — {class_name} — Temporal attention (BiLSTM)\n"
        f"(same attention weights as every other lead — shown against this lead's own waveform)"
    )
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path


def plot_12lead_grid(x_np: np.ndarray, fs: int, class_name: str, save_path: str, lead_names: list) -> str:
    """
    Small-multiples grid: all 12 leads of ONE recording, stacked, sharing
    a time axis — gives the "all leads" view the single-lead figures above
    don't show.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    T = x_np.shape[0]
    time_axis = np.arange(T) / fs
    num_leads = x_np.shape[1]

    fig, axes = plt.subplots(num_leads, 1, figsize=(14, 1.15 * num_leads), sharex=True)
    for i in range(num_leads):
        axes[i].plot(time_axis, x_np[:, i], color="black", linewidth=0.7)
        axes[i].set_ylabel(lead_names[i], rotation=0, labelpad=25, fontsize=10, va="center")
        axes[i].set_yticks([])
        if i < num_leads - 1:
            axes[i].tick_params(labelbottom=False)
    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(f"All 12 leads — {class_name}", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path
