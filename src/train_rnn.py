"""Train a GRU/RNN-style sequence classifier over synthetic or real light-curve sequences.

The RNN is intentionally small and transparent. It accepts a payoff sequence
vector and summarizes the temporal footprint with the last recurrent state.
"""
from __future__ import annotations

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from data_loader import load_lightcurve_sequences
from models_pytorch import RNNClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {"NOT_PLANET": 0, "PLANET": 1, "0": 0, "1": 1}


def _sequence_matrix_from_frame(df):
    rows = []
    y = []
    for _, row in df.iterrows():
        seq = row.get("sequence")
        if seq is None:
            seq = row.get("flux")
        if isinstance(seq, (list, tuple, np.ndarray)):
            arr = np.asarray(seq, dtype=np.float32)
        elif isinstance(seq, str):
            try:
                arr = np.asarray([float(v) for v in seq.split(",")], dtype=np.float32)
            except Exception:
                arr = np.asarray([0.0], dtype=np.float32)
        else:
            arr = np.asarray([float(seq)], dtype=np.float32) if False else np.asarray([0.0], dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.size == 0:
            arr = np.zeros(64, dtype=np.float32)
        if arr.size < 64:
            arr = np.pad(arr, (0, 64 - arr.size), constant_values=0.0)
        if arr.size > 64:
            arr = arr[:64]
        rows.append(arr)
        label = row.get("label")
        y.append(LABEL_MAP.get(str(label), 0))
    X = np.vstack(rows)
    return X.astype(np.float32), np.asarray(y, dtype=np.int64)


def get_data():
    frame, target = load_lightcurve_sequences(data_root="data/lightcurves", max_samples=5000)
    X, y = _sequence_matrix_from_frame(frame.assign(label=target))
    n = len(y)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    train_n = int(n * 0.7)
    val_n = int(n * 0.15)
    train_idx = idx[:train_n]
    val_idx = idx[train_n:train_n + val_n]
    test_idx = idx[train_n + val_n:]
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx]), X.shape[1]


def make_loader(X, y, batch_size=128, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_rnn(epochs=10):
    (X_train, y_train), (X_val, y_val), (X_test, y_test), n_features = get_data()
    os.makedirs("models/sequence", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(len(y_train) / (2.0 * class_counts), dtype=torch.float32).to(DEVICE)
    model = RNNClassifier(input_size=1, hidden_size=32, n_classes=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = make_loader(X_train, y_train, batch_size=128, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=128, shuffle=False)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_n += xb.size(0)
        tr_loss /= tr_n

        model.eval()
        va_loss, va_correct, va_n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                va_loss += loss.item() * xb.size(0)
                va_correct += (logits.argmax(1) == yb).sum().item()
                va_n += xb.size(0)
        va_loss /= va_n
        tr_acc = tr_correct / tr_n
        va_acc = va_correct / va_n

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"epoch {epoch:3d}/{epochs} train_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_loss={va_loss:.4f} acc={va_acc:.3f}")
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), "models/sequence/rnn_final.pt")
    torch.save(model.state_dict(), "models/sequence/rnn_best.pt")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("RNN loss")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("RNN accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("reports/figures/rnn_training_curves.png", dpi=130)
    plt.close(fig)

    return model, history


if __name__ == "__main__":
    train_rnn()
