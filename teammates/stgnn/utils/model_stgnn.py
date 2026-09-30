"""
utils/model_stgnn.py
=====================
The STGNN itself:

    beats -> ResNet1D (per lead)      [Stage A: node features]
          -> edge refinement + GAT    [Stage B: spatial, per beat]
          -> node attention pool      [Stage C: one embedding per beat]
          -> BiGRU                    [Stage D: temporal / rhythm]
          -> temporal attention        [Stage E: context vector]
          -> linear -> 5 logits        [Stage F: multi-label head]

Returns raw logits, not probabilities — pair with BCEWithLogitsLoss (or a
focal variant; SVT has only 482 positives against SB's 12,291, so plain BCE
will underweight it).

Batching
--------
Takes ONE recording per call, unbatched: beats is (n_beats, window_len,
num_leads), exactly as dataset_stgnn's identity_collate hands it over. No
leading batch dim anywhere. This matches the batch_size=1 loader and keeps
the ragged beat-count problem out of the model entirely.

Axis order
----------
The .npz files store (n_beats, window_len, num_leads) — samples before
leads. (The docstrings in dataset_stgnn.py and preprocessing_physionet_stgnn.py
say (n_beats, num_leads, window_len); they are wrong, the files are right.)
This module takes the on-disk order and transposes internally.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config_physionet import PhysioNetConfig as PNConfig
from dataset_stgnn import build_anatomical_adjacency


# ===========================================================================
# Stage A — ResNet1D node-feature extractor
# ===========================================================================

class BasicBlock1d(nn.Module):
    """Pre-activation-style residual block, kernel 7 (wide enough to span a
    QRS at 500 Hz after the stem's 4x downsample)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)

        self.downsample: Optional[nn.Module] = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class ResNet1D(nn.Module):
    """
    Encodes ONE (beat, lead) window into a fixed-length node embedding.

    Weights are shared across all 12 leads and all beats — the leads go in
    as a flattened batch, so lead identity is carried by the graph
    structure in Stage B, not by separate per-lead parameters. This is the
    node feature extractor; it never pools across leads.
    """

    def __init__(self, base_ch: int = 32, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(
            BasicBlock1d(base_ch, base_ch, 1, dropout),
            BasicBlock1d(base_ch, base_ch, 1, dropout),
        )
        self.layer2 = nn.Sequential(
            BasicBlock1d(base_ch, base_ch * 2, 2, dropout),
            BasicBlock1d(base_ch * 2, base_ch * 2, 1, dropout),
        )
        self.layer3 = nn.Sequential(
            BasicBlock1d(base_ch * 2, embed_dim, 2, dropout),
            BasicBlock1d(embed_dim, embed_dim, 1, dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (M, 1, window_len) -> (M, embed_dim), M = n_beats * num_leads."""
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        return self.pool(h).squeeze(-1)


# ===========================================================================
# Step 4 — learned edge refinement over the anatomical prior
# ===========================================================================

class EdgeRefinement(nn.Module):
    """
    Per-beat, gated-residual refinement of the fixed anatomical adjacency:

        A_refined = relu( A_prior + sigmoid(gate) * tanh(MLP([h_i || h_j])) )

    Residual, not replacement — the anatomical prior is the base term and
    the learned delta is added to it. `gate` starts at -3.0, so
    sigmoid(gate) ~= 0.047 at step 0: the graph *is* the Einthoven/Goldberger/
    precordial prior early in training, and the learned term only earns
    influence as the gate opens. Without this, random init would wash the
    anatomy out in the first few hundred steps.

    The delta is symmetrised so the graph stays undirected, and relu keeps
    weights non-negative (they are consumed as a log-prior in the GAT, which
    needs A >= 0).
    """

    def __init__(self, node_dim: int, hidden: int = 32, gate_init: float = -3.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, H: torch.Tensor, A_prior: torch.Tensor) -> torch.Tensor:
        """H: (T, N, d), A_prior: (N, N) -> A: (T, N, N)."""
        T, N, d = H.shape
        hi = H.unsqueeze(2).expand(T, N, N, d)
        hj = H.unsqueeze(1).expand(T, N, N, d)
        delta = torch.tanh(self.mlp(torch.cat([hi, hj], dim=-1)).squeeze(-1))
        delta = 0.5 * (delta + delta.transpose(1, 2))  # undirected
        return F.relu(A_prior.unsqueeze(0) + torch.sigmoid(self.gate) * delta)


# ===========================================================================
# Stage B — dense GAT over the 12 lead-nodes
# ===========================================================================

class DenseGATLayer(nn.Module):
    """
    Multi-head graph attention on one beat snapshot, written against a dense
    (T, N, N) weighted adjacency rather than torch_geometric's edge_index.
    N=12, so dense is both cheaper than building sparse indices per beat and
    keeps the full 12x12 attention map directly available for XAI.

    Edge weights enter as an additive log-prior on the attention logits, so
    post-softmax they scale the coefficients multiplicatively: an
    anatomically strong edge is favoured, a zero-weight edge is masked out.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4,
                 concat: bool = True, dropout: float = 0.1,
                 negative_slope: float = 0.2):
        super().__init__()
        self.heads, self.out_dim, self.concat = heads, out_dim, concat
        self.negative_slope = negative_slope
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        H: (T, N, in_dim), A: (T, N, N) non-negative.
        Returns (out, alpha) with alpha (T, heads, N, N), alpha[t,h,i,j] =
        how much node i attended to node j in beat t.
        """
        T, N, _ = H.shape
        Wh = self.W(H).view(T, N, self.heads, self.out_dim).permute(0, 2, 1, 3)
        e_src = (Wh * self.attn_src.view(1, self.heads, 1, self.out_dim)).sum(-1)
        e_dst = (Wh * self.attn_dst.view(1, self.heads, 1, self.out_dim)).sum(-1)

        logits = F.leaky_relu(
            e_src.unsqueeze(-1) + e_dst.unsqueeze(-2), self.negative_slope
        )                                       # (T, heads, N, N)

        A_h = A.unsqueeze(1)                    # (T, 1, N, N)
        logits = logits + torch.log(A_h + 1e-16)
        logits = logits.masked_fill(A_h <= 0, float("-inf"))

        alpha = torch.softmax(logits, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)   # guard: fully-masked row

        out = torch.einsum("thij,thjd->thid", self.dropout(alpha), Wh)
        if self.concat:
            out = out.permute(0, 2, 1, 3).reshape(T, N, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)
        return out + self.bias, alpha


# ===========================================================================
# Stage C / E — attention pooling (over nodes, then over beats)
# ===========================================================================

class NodeAttentionPool(nn.Module):
    """
    Stage C: collapse the 12 spatially-contextualised lead embeddings of a
    beat into one beat embedding. Attention rather than mean pooling, so the
    per-lead weights are themselves an XAI output ("which lead carried this
    beat"). Quality-rejected leads are masked to zero contribution.
    """

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, H: torch.Tensor, node_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """H: (T, N, d), node_mask: (N,) bool -> ((T, d), (T, N))."""
        s = self.score(H).squeeze(-1)
        s = s.masked_fill(~node_mask.unsqueeze(0), float("-inf"))
        w = torch.softmax(s, dim=-1)
        return torch.einsum("tn,tnd->td", w, H), w


class TemporalAttention(nn.Module):
    """Stage E: additive attention over the BiGRU outputs -> context vector.
    The weights say which beats in the sequence drove the decision."""

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """S: (T, dim) -> ((dim,), (T,))."""
        w = torch.softmax(self.score(S).squeeze(-1), dim=0)
        return (w.unsqueeze(-1) * S).sum(dim=0), w


# ===========================================================================
# Full model
# ===========================================================================

class STGNN(nn.Module):

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
    ):
        super().__init__()
        self.num_leads = num_leads
        self.num_classes = num_classes

        # Stage A
        self.encoder = ResNet1D(resnet_base_ch, node_dim, dropout)

        # Step 4 + Stage B
        self.edge_refine = EdgeRefinement(node_dim, gate_init=edge_gate_init)
        self.gat1 = DenseGATLayer(node_dim, node_dim // gat_heads, gat_heads,
                                  concat=True, dropout=dropout)
        self.gat2 = DenseGATLayer(node_dim, node_dim, gat_heads,
                                  concat=False, dropout=dropout)

        # Stage C
        self.node_pool = NodeAttentionPool(node_dim)

        # Stage D
        self.gru = nn.GRU(node_dim, gru_hidden, num_layers=gru_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if gru_layers > 1 else 0.0)

        # Stage E / F
        self.temporal_attn = TemporalAttention(2 * gru_hidden)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * gru_hidden, num_classes),
        )

        self.register_buffer("A_fixed", build_anatomical_adjacency())

    # ------------------------------------------------------------------
    def _masked_prior(self, lead_valid: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Zero out every edge touching a quality-rejected lead, then force a
        non-zero self-loop on every node. The self-loop matters for more
        than GAT stability: without it an excluded lead's adjacency row is
        all-zero, the GAT masks the whole row to -inf, and the softmax
        returns NaN. With it, an excluded node attends only to itself and is
        then dropped from the pooling anyway.
        """
        A = self.A_fixed.clone()
        if lead_valid is not None:
            m = lead_valid.to(A.dtype)
            A = A * m.unsqueeze(0) * m.unsqueeze(1)
        idx = torch.arange(A.shape[0], device=A.device)
        A[idx, idx] = 1.0
        return A

    def forward(
        self,
        beats: torch.Tensor,
        lead_valid: Optional[torch.Tensor] = None,
        return_xai: bool = False,
    ):
        """
        beats      : (n_beats, window_len, num_leads)  — on-disk axis order
        lead_valid : (num_leads,) bool, or None to treat all leads as valid
        return_xai : also return the attention/edge maps for section 6

        Returns logits (num_classes,), or (logits, xai_dict) if return_xai.
        """
        if beats.dim() != 3:
            raise ValueError(f"expected beats (n_beats, window_len, num_leads), got {tuple(beats.shape)}")
        if beats.shape[-1] != self.num_leads:
            raise ValueError(
                f"last axis of beats must be num_leads={self.num_leads}, got "
                f"{tuple(beats.shape)} — did you pass (n_beats, num_leads, window_len)?"
            )

        T, L, N = beats.shape
        if lead_valid is None:
            lead_valid = torch.ones(N, dtype=torch.bool, device=beats.device)
        lead_valid = lead_valid.to(beats.device).bool()

        # --- Stage A: per (beat, lead) node features -------------------
        x = beats.permute(0, 2, 1).reshape(T * N, 1, L)
        node_feat = self.encoder(x).view(T, N, -1)               # (T, N, node_dim)

        # --- Step 4: hybrid adjacency ----------------------------------
        A_prior = self._masked_prior(lead_valid)                 # (N, N)
        A = self.edge_refine(node_feat, A_prior)                 # (T, N, N)

        # Refinement may propose edges the prior lacks (that is the point),
        # but it must not resurrect edges into rejected leads — re-mask,
        # then restore the self-loops the mask just zeroed.
        m = lead_valid.to(A.dtype)
        A = A * (m.unsqueeze(0) * m.unsqueeze(1)).unsqueeze(0)
        idx = torch.arange(N, device=A.device)
        A[:, idx, idx] = A[:, idx, idx].clamp(min=1e-4)

        # --- Stage B: spatial attention per beat -----------------------
        h, alpha1 = self.gat1(node_feat, A)
        h = F.elu(h)
        h, alpha2 = self.gat2(h, A)
        h = F.elu(h)                                             # (T, N, node_dim)

        # --- Stage C: beat embeddings ----------------------------------
        beat_emb, lead_w = self.node_pool(h, lead_valid)         # (T, node_dim), (T, N)

        # --- Stage D: temporal ----------------------------------------
        seq, _ = self.gru(beat_emb.unsqueeze(0))                 # (1, T, 2*hidden)
        seq = seq.squeeze(0)

        # --- Stage E / F ----------------------------------------------
        context, beat_w = self.temporal_attn(seq)
        logits = self.head(context)                              # (num_classes,)

        if not return_xai:
            return logits

        xai: Dict[str, torch.Tensor] = {
            "edge_weights": A.detach(),        # (T, N, N)  refined adjacency
            "gat_attn_l1": alpha1.detach(),    # (T, heads, N, N)
            "gat_attn_l2": alpha2.detach(),    # (T, heads, N, N)
            "lead_attn": lead_w.detach(),      # (T, N)     which lead, per beat
            "beat_attn": beat_w.detach(),      # (T,)       which beat
            "edge_gate": torch.sigmoid(self.edge_refine.gate).detach(),
        }
        return logits, xai


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================================
# Smoke test
# ===========================================================================

if __name__ == "__main__":
    from dataset_stgnn import make_dataloader

    PNConfig.load_label_vocab()
    torch.manual_seed(PNConfig.RANDOM_SEED)

    model = STGNN(num_classes=PNConfig.NUM_CLASSES).to(PNConfig.DEVICE)
    print(f"Device            : {PNConfig.DEVICE}")
    print(f"Trainable params  : {count_parameters(model):,}")
    print(f"Edge gate at init : {torch.sigmoid(model.edge_refine.gate).item():.4f}")

    loader = make_dataloader("train", shuffle=False)
    batch = next(iter(loader))
    beats = batch["beats"].to(PNConfig.DEVICE)
    lead_valid = batch["lead_valid"].to(PNConfig.DEVICE)

    print(f"\nRecord            : {batch['record_id']}")
    print(f"beats in          : {tuple(beats.shape)}  (n_beats, window_len, leads)")
    print(f"lead_valid        : {lead_valid.sum().item()}/{PNConfig.NUM_LEADS} valid")

    model.eval()
    with torch.no_grad():
        logits, xai = model(beats, lead_valid, return_xai=True)

    print(f"\nlogits            : {tuple(logits.shape)}")
    print(f"probs             : "
          f"{dict(zip(PNConfig.CLASS_NAMES, torch.sigmoid(logits).round(decimals=3).tolist()))}")
    print(f"true label        : "
          f"{[n for n, v in zip(PNConfig.CLASS_NAMES, batch['label'].tolist()) if v]}")
    print("\nXAI tensors:")
    for k, v in xai.items():
        print(f"  {k:14s} {tuple(v.shape)}")
    print(f"\nbeat_attn sums to {xai['beat_attn'].sum().item():.4f} "
          f"(should be 1.0)")
    print(f"lead_attn row sums {xai['lead_attn'].sum(-1)[:3].tolist()} "
          f"(should all be 1.0)")
    assert torch.isfinite(logits).all(), "non-finite logits"
    print("\nOK")
