"""
utils/model_stgnn_rr.py
========================
STGNN + R-R interval features. New file; model_stgnn.py is imported, not
modified — every building block (ResNet1D, EdgeRefinement, DenseGATLayer,
NodeAttentionPool, TemporalAttention) is reused unchanged, so the spatial half
of the architecture is provably identical to the version you already trained.

What changes, and why
---------------------
Diagnostic result on 2,992 training records: a bare heart-rate threshold scores
0.9791 on SB/SR/ST, beating the trained network on all three (SB .985 vs .959,
SR .973 vs .892, ST .973 vs .930). AFIB's R-R variability is 0.222 against
0.014-0.021 for every regular rhythm. The information is trivially available
and the current model does not have it.

Two injection points, deliberately:

  per beat     the 6-dim R-R vector is encoded and concatenated to that beat's
               pooled GAT embedding before the BiGRU, so the recurrent layer
               sees rhythm evolving beat to beat (this is what carries AFIB's
               beat-to-beat irregularity).

  per record   the three record-level features (median R-R, CV, heart rate) are
               encoded again and concatenated to the temporal-attention context
               vector, immediately before the classifier. Rate is a
               record-level property and the SB/SR/ST decision is a threshold
               on it; routing it through 12 BiGRU steps and an attention
               weighting first would only give the network a chance to lose it.

An auxiliary cross-entropy head is NOT here — that lives in the loss, in
train_stgnn_rr.py, so this module still emits exactly 5 sigmoid logits and stays
compatible with evaluate_stgnn.py and calibrate_stgnn.py.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config_physionet import PhysioNetConfig as PNConfig
from dataset_rr import N_RR_FEATURES
from dataset_stgnn import build_anatomical_adjacency
from model_stgnn import (
    DenseGATLayer,
    EdgeRefinement,
    NodeAttentionPool,
    ResNet1D,
    TemporalAttention,
    count_parameters,
)


class STGNN_RR(nn.Module):

    def __init__(
        self,
        num_classes: int = PNConfig.NUM_CLASSES,
        num_leads: int = PNConfig.NUM_LEADS,
        node_dim: int = 128,
        resnet_base_ch: int = 32,
        gat_heads: int = 4,
        gru_hidden: int = 128,
        gru_layers: int = 1,
        dropout: float = 0.1,
        edge_gate_init: float = -3.0,
        rr_dim: int = 32,
        rr_summary_dim: int = 16,
    ):
        super().__init__()
        self.num_leads = num_leads
        self.num_classes = num_classes

        # --- unchanged spatial stack ---------------------------------
        self.encoder = ResNet1D(resnet_base_ch, node_dim, dropout)
        self.edge_refine = EdgeRefinement(node_dim, gate_init=edge_gate_init)
        self.gat1 = DenseGATLayer(node_dim, node_dim // gat_heads, gat_heads,
                                  concat=True, dropout=dropout)
        self.gat2 = DenseGATLayer(node_dim, node_dim, gat_heads,
                                  concat=False, dropout=dropout)
        self.node_pool = NodeAttentionPool(node_dim)

        # --- new: R-R encoders ---------------------------------------
        self.rr_encoder = nn.Sequential(
            nn.Linear(N_RR_FEATURES, rr_dim), nn.ReLU(inplace=True),
            nn.Linear(rr_dim, rr_dim), nn.ReLU(inplace=True),
        )
        # median R-R, CV, heart rate — the record-level triple.
        self.rr_summary = nn.Sequential(
            nn.Linear(3, rr_summary_dim), nn.ReLU(inplace=True),
        )

        self.gru = nn.GRU(node_dim + rr_dim, gru_hidden, num_layers=gru_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if gru_layers > 1 else 0.0)
        self.temporal_attn = TemporalAttention(2 * gru_hidden)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * gru_hidden + rr_summary_dim, num_classes),
        )

        self.register_buffer("A_fixed", build_anatomical_adjacency())

    # ------------------------------------------------------------------
    def _masked_prior(self, lead_valid: Optional[torch.Tensor]) -> torch.Tensor:
        A = self.A_fixed.clone()
        if lead_valid is not None:
            m = lead_valid.to(A.dtype)
            A = A * m.unsqueeze(0) * m.unsqueeze(1)
        idx = torch.arange(A.shape[0], device=A.device)
        A[idx, idx] = 1.0
        return A

    def forward(self, beats: torch.Tensor, rr: torch.Tensor,
                lead_valid: Optional[torch.Tensor] = None,
                return_xai: bool = False):
        """
        beats      : (n_beats, window_len, num_leads)
        rr         : (n_beats, 6)   from dataset_rr.compute_rr_features
        lead_valid : (num_leads,) bool or None
        """
        if beats.shape[-1] != self.num_leads:
            raise ValueError(f"beats last axis must be num_leads={self.num_leads}, "
                             f"got {tuple(beats.shape)}")
        if rr.shape[0] != beats.shape[0]:
            raise ValueError(f"rr has {rr.shape[0]} rows but there are "
                             f"{beats.shape[0]} beats")

        T, L, N = beats.shape
        if lead_valid is None:
            lead_valid = torch.ones(N, dtype=torch.bool, device=beats.device)
        lead_valid = lead_valid.to(beats.device).bool()

        # Stage A
        x = beats.permute(0, 2, 1).reshape(T * N, 1, L)
        node_feat = self.encoder(x).view(T, N, -1)

        # Step 4
        A_prior = self._masked_prior(lead_valid)
        A = self.edge_refine(node_feat, A_prior)
        m = lead_valid.to(A.dtype)
        A = A * (m.unsqueeze(0) * m.unsqueeze(1)).unsqueeze(0)
        idx = torch.arange(N, device=A.device)
        A[:, idx, idx] = A[:, idx, idx].clamp(min=1e-4)

        # Stage B
        h, alpha1 = self.gat1(node_feat, A)
        h = F.elu(h)
        h, alpha2 = self.gat2(h, A)
        h = F.elu(h)

        # Stage C + per-beat R-R
        beat_emb, lead_w = self.node_pool(h, lead_valid)          # (T, node_dim)
        beat_emb = torch.cat([beat_emb, self.rr_encoder(rr)], dim=-1)

        # Stage D / E
        seq, _ = self.gru(beat_emb.unsqueeze(0))
        context, beat_w = self.temporal_attn(seq.squeeze(0))

        # Stage F, with the record-level rate/irregularity triple alongside
        summary = self.rr_summary(rr[0, 3:6])
        logits = self.head(torch.cat([context, summary], dim=-1))

        if not return_xai:
            return logits

        xai: Dict[str, torch.Tensor] = {
            "edge_weights": A.detach(),
            "gat_attn_l1": alpha1.detach(),
            "gat_attn_l2": alpha2.detach(),
            "lead_attn": lead_w.detach(),
            "beat_attn": beat_w.detach(),
            "edge_gate": torch.sigmoid(self.edge_refine.gate).detach(),
            "rr": rr.detach(),
        }
        return logits, xai


if __name__ == "__main__":
    from dataset_rr import build_rr_loader

    PNConfig.load_label_vocab()
    torch.manual_seed(PNConfig.RANDOM_SEED)

    model = STGNN_RR(num_classes=PNConfig.NUM_CLASSES).to(PNConfig.DEVICE)
    print(f"params : {count_parameters(model):,}")

    batch = next(iter(build_rr_loader("train", limit=4, num_workers=0)))
    beats = batch["beats"].to(PNConfig.DEVICE)
    rr = batch["rr"].to(PNConfig.DEVICE)
    lead_valid = batch["lead_valid"].to(PNConfig.DEVICE)

    print(f"record : {batch['record_id']}")
    print(f"beats  : {tuple(beats.shape)}   rr {tuple(rr.shape)}")

    model.eval()
    with torch.no_grad():
        logits, xai = model(beats, rr, lead_valid, return_xai=True)
    print(f"logits : {tuple(logits.shape)}")
    print(f"probs  : {dict(zip(PNConfig.CLASS_NAMES, torch.sigmoid(logits).round(decimals=3).tolist()))}")
    print(f"true   : {[n for n, v in zip(PNConfig.CLASS_NAMES, batch['label'].tolist()) if v]}")
    assert torch.isfinite(logits).all()
    print("OK")
