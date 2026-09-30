"""
explain_recording.py
=====================
Generates a full explainability report for ONE ECG recording: prediction,
confidence, lead-attention weights, temporal-attention trace, and
Grad-CAM++ saliency (both per-lead and aggregated).

This is the "explain a single recording" building block the web app's
/explain endpoint will wrap directly — everything here is already
structured as (numbers in, numbers + one PNG out), no plotting-only logic
mixed into the computation.

Usage
-----
    python explain_recording.py --model cnn_bilstm2 --test_index 0
    python explain_recording.py --model cnn_bilstm2 --test_index 0 --lead II
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")

from config_physionet import PhysioNetConfig as PNConfig
from train_physionet import MODEL_REGISTRY, build_model
from trainer import set_seed
from xai.gradcam import GradCAMPlusPlus

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def load_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def explain_single_recording(model, x: torch.Tensor, device: torch.device):
    """
    Core computation, reusable by the web app.

    Parameters
    ----------
    x : Tensor, shape (1, T, num_leads)

    Returns
    -------
    dict with keys: predicted_class, confidence, probs, per_lead_cam,
    aggregated_cam, lead_weights, temporal_attention
    """
    x = x.to(device)
    model = model.to(device).eval()

    # --- Prediction + confidence + temporal attention (no grad needed) ---
    with torch.no_grad():
        logits, lead_weights_plain, temporal_attn, _ = model(x, return_lead_weights=True)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        predicted_class = int(probs.argmax())
        confidence = float(probs[predicted_class])
        temporal_attn = temporal_attn.squeeze(0).squeeze(-1).cpu().numpy()  # (T_lstm,)

    # --- Grad-CAM++ (needs its own forward+backward pass, with grad) ---
    gradcam = GradCAMPlusPlus(model)
    try:
        per_lead_cam, aggregated_cam, gradcam_class, lead_weights = gradcam.generate(x, target_class=predicted_class)
    finally:
        gradcam.remove_hooks()

    assert gradcam_class == predicted_class, "Grad-CAM++ explained a different class than was predicted."

    # Upsample temporal attention to full signal length for plotting
    # against the raw waveform (same approach as the lead-attention /
    # temporal-attention visualisation scripts).
    T = x.shape[1]
    x_old = np.linspace(0, T - 1, len(temporal_attn))
    x_new = np.arange(T)
    temporal_attn_upsampled = np.interp(x_new, x_old, temporal_attn)
    temporal_attn_upsampled -= temporal_attn_upsampled.min()
    if temporal_attn_upsampled.max() > 0:
        temporal_attn_upsampled /= temporal_attn_upsampled.max()

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probs": probs,
        "per_lead_cam": per_lead_cam.numpy(),          # (L, T)
        "aggregated_cam": aggregated_cam.numpy(),        # (T,)
        "lead_weights": lead_weights.numpy(),            # (L,)
        "temporal_attention": temporal_attn_upsampled,   # (T,)
    }


def plot_explanation(x_np, result, class_names, lead_names, save_path, true_label=None, display_lead="II"):
    T = x_np.shape[0]
    lead_idx = lead_names.index(display_lead) if display_lead in lead_names else 1
    signal = x_np[:, lead_idx]

    pred_name = class_names[result["predicted_class"]]
    conf = result["confidence"]
    title_bits = f"Predicted: {pred_name}  (confidence {conf:.1%})"
    if true_label is not None:
        title_bits += f"   |   True: {class_names[true_label]}"

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), gridspec_kw={"height_ratios": [2, 1, 1, 2]})

    # Panel 1: raw signal + aggregated Grad-CAM++ overlay
    axes[0].plot(signal, color="black", linewidth=1.0)
    ax0b = axes[0].twinx()
    ax0b.fill_between(np.arange(T), 0, result["aggregated_cam"], color="red", alpha=0.3)
    ax0b.set_ylim(0, 1.05)
    ax0b.set_ylabel("Grad-CAM++ (aggregated)", color="red")
    axes[0].set_title(f"Lead {display_lead} — Grad-CAM++ saliency\n{title_bits}")
    axes[0].set_ylabel("Amplitude")

    # Panel 2: temporal attention overlay
    axes[1].plot(signal, color="black", linewidth=0.8)
    ax1b = axes[1].twinx()
    ax1b.fill_between(np.arange(T), 0, result["temporal_attention"], color="blue", alpha=0.3)
    ax1b.set_ylim(0, 1.05)
    ax1b.set_ylabel("Temporal attention", color="blue")
    axes[1].set_title(f"Lead {display_lead} — Temporal attention (BiLSTM)")

    # Panel 3: lead-attention bar chart
    axes[2].bar(lead_names, result["lead_weights"])
    axes[2].set_title("Lead-attention weights (which of the 12 leads mattered most)")
    axes[2].set_ylabel("Weight")

    # Panel 4: per-lead Grad-CAM++ heatmap
    im = axes[3].imshow(result["per_lead_cam"], aspect="auto", cmap="hot",
                          extent=[0, T, len(lead_names), 0])
    axes[3].set_yticks(np.arange(len(lead_names)) + 0.5)
    axes[3].set_yticklabels(lead_names)
    axes[3].set_xlabel("Time (samples)")
    axes[3].set_title("Grad-CAM++ saliency, per lead (enabled by the vectorized architecture)")
    fig.colorbar(im, ax=axes[3], label="Saliency")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path


def parse_args():
    parser = argparse.ArgumentParser(description="Explain a single ECG recording.")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test_index", type=int, default=0, help="Index into the test set to explain.")
    parser.add_argument("--lead", type=str, default="II", help="Which lead to display the waveform for.")
    parser.add_argument("--seed", type=int, default=PNConfig.RANDOM_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    PNConfig.create_output_dirs()
    PNConfig.load_label_vocab()
    set_seed(args.seed)

    X_test = np.load(os.path.join(PNConfig.PROCESSED_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(PNConfig.PROCESSED_PATH, "y_test.npy"))

    x_np = X_test[args.test_index]
    true_label = int(y_test[args.test_index])
    x = torch.from_numpy(x_np).float().unsqueeze(0)

    model = build_model(args.model)
    checkpoint_path = args.checkpoint or os.path.join(
        PNConfig.CHECKPOINT_PATH, f"physionet_{args.model}_best_model.pth"
    )
    model = load_checkpoint(model, checkpoint_path, PNConfig.DEVICE)

    logger.info("Explaining test sample %d (true label: %s)...", args.test_index, PNConfig.CLASS_NAMES[true_label])
    result = explain_single_recording(model, x, PNConfig.DEVICE)

    logger.info("Predicted: %s  (confidence %.1f%%)",
                PNConfig.CLASS_NAMES[result["predicted_class"]], result["confidence"] * 100)
    logger.info("Correct: %s", result["predicted_class"] == true_label)

    save_dir = os.path.join(PNConfig.PLOTS_PATH, "explanations")
    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, f"explanation_test{args.test_index}.png")
    plot_explanation(
        x_np, result, PNConfig.CLASS_NAMES, PNConfig.LEAD_NAMES, png_path,
        true_label=true_label, display_lead=args.lead,
    )
    logger.info("Explanation figure saved -> %s", png_path)

    # Save the raw numbers too, so this is reusable without re-plotting
    json_path = os.path.join(save_dir, f"explanation_test{args.test_index}.json")
    with open(json_path, "w") as fh:
        json.dump({
            "test_index": args.test_index,
            "true_label": PNConfig.CLASS_NAMES[true_label],
            "predicted_class": PNConfig.CLASS_NAMES[result["predicted_class"]],
            "confidence": result["confidence"],
            "probs": {PNConfig.CLASS_NAMES[i]: float(p) for i, p in enumerate(result["probs"])},
            "lead_weights": {PNConfig.LEAD_NAMES[i]: float(w) for i, w in enumerate(result["lead_weights"])},
        }, fh, indent=2)
    logger.info("Explanation data saved -> %s", json_path)


if __name__ == "__main__":
    main()