from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from config_physionet import PhysioNetConfig as Config
from models.cnn2 import CNNBaseline


class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = Config.NUM_CLASSES,
        hidden_size: int = Config.LSTM_HIDDEN_SIZE,
        num_layers: int = Config.LSTM_NUM_LAYERS,
        dropout: float = Config.FC_DROPOUT,
        num_leads: int = Config.NUM_LEADS,
        use_lead_attention: bool = True,
        use_temporal_attention: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_temporal_attention = use_temporal_attention

        self.cnn_backbone = CNNBaseline(num_classes=num_classes, dropout=0.0, num_leads=num_leads,
                                         use_lead_attention=use_lead_attention)
        cnn_out_dim = Config.CNN_FILTERS[-1]

        self.bilstm = nn.LSTM(
            input_size=cnn_out_dim, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        self.post_lstm_dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size * 2, num_classes))
        self._init_weights()

    def _init_weights(self):
        for name, param in self.bilstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for m in self.attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, return_lead_weights: bool = False):
        fused_features, lead_weights, cnn_features = self.cnn_backbone.extract_features(x)
        lstm_outputs, (h_n, _c_n) = self.bilstm(fused_features)
        lstm_outputs = self.post_lstm_dropout(lstm_outputs)

        if self.use_temporal_attention:
            attention_scores = self.attention(lstm_outputs)
            attention_weights = torch.softmax(attention_scores, dim=1)
            context_vector = torch.sum(attention_weights * lstm_outputs, dim=1)
        else:
            # Ablation: no temporal attention. Use the BiLSTM's final hidden
            # state (forward + backward concatenated) instead of an
            # attention-weighted sum over all timesteps — the standard,
            # simplest alternative pooling strategy, isolating what temporal
            # attention specifically contributes over "just use the last
            # hidden state," holding everything else fixed.
            h_forward = h_n[-2, :, :]   # last layer, forward direction
            h_backward = h_n[-1, :, :]  # last layer, backward direction
            context_vector = torch.cat([h_forward, h_backward], dim=1)
            attention_weights = None

        logits = self.classifier(context_vector)
        if return_lead_weights:
            return logits, lead_weights, attention_weights, cnn_features
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CNNBiLSTM(num_classes=5, num_leads=12)
    dummy = torch.randn(4, 5000, 12)
    logits = model(dummy)
    assert logits.shape == (4, 5)
    print("OK — cnn_bilstm2.py:", logits.shape, f"{model.count_parameters():,} params")