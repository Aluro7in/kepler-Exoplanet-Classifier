"""
models_pytorch.py
Two neural architectures for the same tabular feature matrix:

1. MLPClassifier   - the feed-forward network from the master prompt (§5).
2. TabTransformer   - a compact FT-Transformer-style model: every numeric
                       feature becomes a "token" via its own linear
                       embedding, a learned [CLS] token is prepended, the
                       sequence goes through a standard TransformerEncoder,
                       and the CLS output is classified. This is the
                       "transformer over tabular data" architecture used in
                       the ML-DL research request.
"""
import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x, return_embedding=False):
        emb = self.net(x)
        logits = self.head(emb)
        if return_embedding:
            return logits, emb
        return logits

class CNN1DClassifier(nn.Module):
    """1D CNN that treats a light-curve-like sequence as a local pattern.
    The first layer learns short transit/flux-shape motifs, the later layers
    compress the sequence into a classifiable representation."""
    def __init__(self, n_features: int = 64, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.seq_len = n_features // 4
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * self.seq_len, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, return_embedding=False):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        emb = self.net(x)
        logits = self.fc(emb)
        if return_embedding:
            return logits, emb
        return logits


class RNNClassifier(nn.Module):
    """GRU-based sequence classifier for the light-curve reconstruction.
    The recurrence layer accumulates temporal context from flux values and
    exposes an aggregate state for the binary exoplanet/no-exoplanet decision."""
    def __init__(self, input_size: int = 1, hidden_size: int = 32, n_classes: int = 2, n_layers: int = 2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes),
        )

    def forward(self, x, return_embedding=False):
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        emb = out[:, -1, :]
        logits = self.head(emb)
        if return_embedding:
            return logits, emb
        return logits


class FeatureTokenizer(nn.Module):
    """Turns each scalar feature into a d_model-dim token via its own
    per-feature linear layer (this is the key FT-Transformer trick)."""
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))

    def forward(self, x):
        # x: (batch, n_features) -> (batch, n_features, d_model)
        return x.unsqueeze(-1) * self.weight + self.bias

class TabTransformer(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 2,
                 d_model: int = 32, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.15):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, return_embedding=False):
        tokens = self.tokenizer(x)                      # (B, F, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)   # (B, 1, D)
        seq = torch.cat([cls, tokens], dim=1)            # (B, F+1, D)
        encoded = self.encoder(seq)
        cls_out = self.norm(encoded[:, 0, :])             # (B, D)
        logits = self.head(cls_out)
        if return_embedding:
            return logits, cls_out
        return logits
