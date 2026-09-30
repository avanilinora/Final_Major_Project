"""
xai/gradcam.py
===============
Grad-CAM++ for the TrustECG CNN-BiLSTM model, hooking the shared lead
encoder's final convolutional block.

*** Design note: why this differs from the original single-lead-loop version ***
The original (pre-vectorization) architecture called `lead_encoder` once
PER LEAD inside a 12-iteration Python loop. A `register_forward_hook` /
`register_full_backward_hook` on a shared module that's invoked multiple
times per forward pass only ever holds the state from ONE of those
invocations (a plain assignment in the hook callback overwrites on every
call) — so a Grad-CAM++ implementation hooked that way was, at best,
silently explaining only one lead's activations, not a genuine
multi-lead explanation, and at worst mixing mismatched forward/backward
invocations if hook firing order didn't line up.

This version hooks the SAME module, but because `extract_features` (see
models/cnn2.py) now runs all 12 leads through the shared encoder in a
single batched call, the hook fires exactly ONCE per forward pass and
captures all leads' activations together, shape (B*L, C, T'). This is
verified defensively below (a call counter that must equal 1) rather than
just assumed.

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

        # *** CUDA/cuDNN fix (do not remove) ***
        # cuDNN's fused LSTM kernel only retains the internal workspace
        # needed for backward() when the module is in train() mode. In
        # eval() mode — which is correct here, since we don't want
        # Dropout/BatchNorm behaving stochastically while explaining a
        # prediction — cuDNN discards that workspace, and calling
        # backward() later raises:
        #   RuntimeError: cudnn RNN backward can only be called in training mode
        # This does NOT mean the model needs to be in train() mode; it
        # means cuDNN's specific fused kernel can't be used here. Disabling
        # cuDNN for just this forward+backward pair makes PyTorch fall back
        # to its plain (non-fused) LSTM implementation, which has no such
        # restriction — model.eval() behavior (dropout off, BatchNorm using
        # running stats) is completely unaffected either way. Only shows up
        # on CUDA; CPU-only environments never hit this code path.
        with torch.backends.cudnn.flags(enabled=False):
            logits, lead_weights, _attn_weights, _cnn_features = self.model(x, return_lead_weights=True)

            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())

            score = logits[0, target_class]
            score.backward(retain_graph=True)

        # Defensive check: the whole point of hooking the vectorized
        # encoder is that it's called exactly once per forward pass. If
        # this ever fires more than once, something about the model
        # architecture changed and this CAM implementation's assumptions
        # no longer hold — fail loudly rather than silently explain the
        # wrong thing.
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