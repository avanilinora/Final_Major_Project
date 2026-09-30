from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from config_physionet import PhysioNetConfig as Config


class LeadAttention(nn.Module):
    def __init__(self, feature_dim: int, num_leads: int = 2, use_attention: bool = True):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.num_leads = num_leads
        # Ablation flag: when False, fuse leads by simple mean instead of a
        # learned attention weighting. Used for the "no lead attention"
        # ablation variant — isolates the contribution of the lead-attention
        # mechanism itself, holding everything else in the architecture fixed.
        self.use_attention = use_attention

    def forward(self, x: Tensor):
        if not self.use_attention:
            # Uniform weighting = plain mean across leads.
            L = x.shape[-1]
            weights = torch.full((x.shape[0], L), 1.0 / L, device=x.device, dtype=x.dtype)
            fused = x.mean(dim=-1)
            return fused, weights
        lead_repr = x.mean(dim=1).permute(0, 2, 1)
        scores = self.attention(lead_repr).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        fused = (x * weights.unsqueeze(1).unsqueeze(1)).sum(dim=-1)
        return fused, weights


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same", bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class LeadEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = CNNBlock(1, Config.CNN_FILTERS[0], 5, pool=True)
        self.block2 = CNNBlock(Config.CNN_FILTERS[0], Config.CNN_FILTERS[1], 5, pool=True)
        self.block3 = CNNBlock(Config.CNN_FILTERS[1], Config.CNN_FILTERS[2], 3, pool=False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.permute(0, 2, 1)


class CNNBaseline(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES, dropout=Config.FC_DROPOUT, num_leads: int = Config.NUM_LEADS,
                 use_lead_attention: bool = True):
        super().__init__()
        self.num_leads = num_leads
        self.lead_encoder = LeadEncoder()
        self.lead_attention = LeadAttention(feature_dim=Config.CNN_FILTERS[-1], num_leads=num_leads,
                                             use_attention=use_lead_attention)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(Config.CNN_FILTERS[-1], num_classes))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def extract_features(self, x: Tensor):
        # Vectorized: (B,T,L) -> (B*L,1,T) through the shared encoder in one
        # batched call instead of a 12-iteration Python loop. Verified
        # numerically identical to the loop version at eval time.
        B, T, L = x.shape
        x_reshaped = x.permute(0, 2, 1).reshape(B * L, 1, T)
        feat = self.lead_encoder(x_reshaped)
        Tp, C = feat.shape[1], feat.shape[2]
        features = feat.view(B, L, Tp, C).permute(0, 2, 3, 1)
        fused_features, weights = self.lead_attention(features)
        return fused_features, weights, features

    def extract_raw_features(self, x: Tensor):
        B, T, L = x.shape
        x_reshaped = x.permute(0, 2, 1).reshape(B * L, 1, T)
        feat = self.lead_encoder(x_reshaped)
        Tp, C = feat.shape[1], feat.shape[2]
        return feat.view(B, L, Tp, C).permute(0, 2, 3, 1)

    def forward(self, x: Tensor):
        fused_features, _, _ = self.extract_features(x)
        fused_features = fused_features.permute(0, 2, 1)
        pooled = self.gap(fused_features).squeeze(-1)
        return self.classifier(pooled)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CNNBaseline(num_classes=5, num_leads=12)
    dummy = torch.randn(4, 5000, 12)
    logits = model(dummy)
    assert logits.shape == (4, 5)
    print("OK — cnn2.py:", logits.shape, f"{model.count_parameters():,} params")