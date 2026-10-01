"""
xai/gradcam.py
===============
Grad-CAM++ for the TrustECG CNN-BiLSTM model, hooking the shared lead
encoder's final convolutional block.



Produces:
  - A per-lead, per-timestep saliency map, shape (L, T) — richer than a
    single aggregated trace, and directly comparable to the lead-attention
    weights (Section 5.1 of the project report).
  - A single aggregated (T,) trace, weighted by the model's own learned
    lead-attention weights, for a simple one-panel overlay plot.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class GradCAMPlusPlus:

    def __init__(self, model):
        self.model = model
        self.model.eval()

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._forward_call_count = 0
        self._backward_call_count = 0

        # The shared, single-instance conv block used for every lead.
        self.target_layer = self.model.cnn_backbone.lead_encoder.block3.conv

        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        self._forward_call_count += 1

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        self._backward_call_count += 1

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor, shape (1, T, num_leads)
            A single ECG recording (batch size must be 1 — this method is
            designed for per-recording explanation, not batched CAMs).
        target_class : int, optional
            Class to explain. Defaults to the model's own predicted class.

        Returns
        -------
        per_lead_cam : Tensor, shape (num_leads, T)
            Min-max normalised saliency per lead, upsampled to the full
            signal length.
        aggregated_cam : Tensor, shape (T,)
            per_lead_cam combined via the model's own lead-attention
            weights, then re-normalised to [0, 1].
        predicted_class : int
        lead_weights : Tensor, shape (num_leads,)
            The model's learned lead-attention weights for this recording
            (returned alongside the CAM so a caller can plot both together
            without a second forward pass).
        """
        if x.shape[0] != 1:
            raise ValueError(f"generate() expects batch size 1, got {x.shape[0]}.")

        self.model.zero_grad()
        self._forward_call_count = 0
        self._backward_call_count = 0

       
        with torch.backends.cudnn.flags(enabled=False):
            logits, lead_weights, _attn_weights, _cnn_features = self.model(x, return_lead_weights=True)

            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())

            score = logits[0, target_class]
            score.backward(retain_graph=True)

       
        if self._forward_call_count != 1 or self._backward_call_count != 1:
            raise RuntimeError(
                f"Expected the hooked layer to fire exactly once per "
                f"forward/backward pass (vectorized architecture), but "
                f"got {self._forward_call_count} forward call(s) and "
                f"{self._backward_call_count} backward call(s). Grad-CAM++ "
                f"assumptions are violated — do not trust a CAM computed "
                f"this way."
            )

        activations = self.activations  # (L, C, T')  [batch=1, so B*L == L]
        gradients = self.gradients      # (L, C, T')

        num_leads = x.shape[2]
        if activations.shape[0] != num_leads:
            raise RuntimeError(
                f"Expected {num_leads} activation rows (one per lead, "
                f"batch=1), got {activations.shape[0]}."
            )

        # --- Grad-CAM++ weighting ---
        grads2 = gradients.pow(2)
        grads3 = gradients.pow(3)

        denominator = (2 * grads2 + activations * grads3).sum(dim=2, keepdim=True)
        denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))

        alpha = grads2 / (denominator + 1e-8)
        positive_gradients = F.relu(gradients)
        weights = (alpha * positive_gradients).sum(dim=2)  # (L, C)

        cam = (weights.unsqueeze(-1) * activations).sum(dim=1)  # (L, T')
        cam = F.relu(cam)

        # Upsample each lead's CAM to the full signal length.
        T = x.shape[1]
        cam = cam.unsqueeze(1)  # (L, 1, T')
        cam = F.interpolate(cam, size=T, mode="linear", align_corners=False)
        cam = cam.squeeze(1)  # (L, T)

        # Per-lead min-max normalisation (each lead's trace independently
        # scaled to [0,1], so a flat/uninformative lead doesn't get
        # artificially inflated by a global normalisation).
        cam_min = cam.min(dim=1, keepdim=True).values
        cam_max = cam.max(dim=1, keepdim=True).values
        per_lead_cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Aggregate across leads using the model's own lead-attention
        # weights — leads the model itself considered more important
        # contribute more to the combined trace.
        lw = lead_weights.squeeze(0)  # (L,)
        aggregated = (per_lead_cam * lw.unsqueeze(1)).sum(dim=0)  # (T,)
        aggregated = aggregated - aggregated.min()
        aggregated = aggregated / (aggregated.max() + 1e-8)

        return (
            per_lead_cam.detach().cpu(),
            aggregated.detach().cpu(),
            target_class,
            lw.detach().cpu(),
        )

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass
