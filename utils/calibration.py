"""
utils/calibration.py
=====================
Confidence calibration via temperature scaling (Guo et al., 2017,
"On Calibration of Modern Neural Networks").

A neural network's raw softmax confidence is frequently a poor estimate of
its true accuracy — usually overconfident, meaning a "90% confident"
prediction is right less than 90% of the time. This matters directly for a
clinical-facing tool like TrustECG: a confidence score that can't be
trusted at face value is worse than no confidence score at all, since it
invites false reassurance.

Temperature scaling is the simplest effective fix: divide the model's
logits by a single learned scalar T > 1 before the softmax. This does NOT
change which class the model predicts (dividing all logits by the same
constant doesn't change their ranking, so accuracy is completely
unaffected) — it only reshapes how confident the model is allowed to
sound. T is fit by minimizing negative log-likelihood on the VALIDATION
set (never the test set, to avoid calibrating against the same data used
to report final numbers), using a tiny 1-parameter optimisation.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class TemperatureScaler(nn.Module):
    """
    Wraps a trained, frozen classification model and applies a single
    learned temperature to its logits. The wrapped model's own weights
    are never updated — only `self.temperature` is trained.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # Start at T=1 (no change) — this is what "uncalibrated" looks like.
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: Tensor) -> Tensor:
        logits = self.model(x)
        logits = logits[0] if isinstance(logits, tuple) else logits
        return self.scale(logits)

    def scale(self, logits: Tensor) -> Tensor:
        return logits / self.temperature.clamp(min=1e-3)


@torch.no_grad()
def _collect_logits(model: nn.Module, loader, device: torch.device) -> Tuple[Tensor, Tensor]:
    """Run the model once over a loader and collect raw logits + labels."""
    model.eval()
    all_logits, all_labels = [], []
    for samples, labels in loader:
        samples = samples.to(device)
        output = model(samples)
        logits = output[0] if isinstance(output, tuple) else output
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(
    model: nn.Module,
    val_loader,
    device: torch.device,
    max_iter: int = 100,
    lr: float = 0.01,
) -> float:
    """
    Fit a single temperature parameter on the validation set.

    Parameters
    ----------
    model : trained, already-loaded classification model (frozen — its
        weights are not touched here).
    val_loader : DataLoader over the VALIDATION set. Do not pass the test
        loader here — fitting the temperature on test data would make the
        final calibration numbers optimistic in exactly the way this
        module exists to avoid.

    Returns
    -------
    float
        The fitted temperature. T > 1 means the model was overconfident
        (the common case) and its confidences are being softened; T < 1
        would mean it was underconfident.
    """
    model = model.to(device)
    logits, labels = _collect_logits(model, val_loader, device)
    logits, labels = logits.to(device), labels.to(device)

    scaler = TemperatureScaler(model).to(device)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)
    nll_criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll_criterion(scaler.scale(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    T = float(scaler.temperature.detach().cpu().clamp(min=1e-3).item())
    logger.info("Fitted temperature: T = %.4f", T)
    return T


@torch.no_grad()
def get_calibrated_probs(model: nn.Module, loader, device: torch.device, temperature: float = 1.0):
    """
    Run the model over `loader` and return (y_true, y_pred, probs) with
    the given temperature applied. temperature=1.0 reproduces the
    model's raw, uncalibrated softmax output — pass the T from
    fit_temperature() to get calibrated probabilities instead.
    """
    model = model.to(device).eval()
    all_true, all_pred, all_prob = [], [], []
    for samples, labels in loader:
        samples = samples.to(device)
        output = model(samples)
        logits = output[0] if isinstance(output, tuple) else output
        probs = F.softmax(logits / max(temperature, 1e-3), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_true.extend(labels.numpy().tolist())
        all_pred.extend(preds.tolist())
        all_prob.append(probs)

    import numpy as np
    return np.array(all_true), np.array(all_pred), np.vstack(all_prob)