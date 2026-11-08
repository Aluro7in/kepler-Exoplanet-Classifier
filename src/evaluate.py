"""
evaluate.py
Loads every saved model (4 classical ML + MLP + TabTransformer) and scores
them all on the same held-out test set, producing one comparison table.
"""
import os
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from data_loader import load_labeled_tce
from preprocessing import build_feature_matrix
from split import grouped_train_val_test_split
from models_pytorch import MLPClassifier, TabTransformer
from experiment_tracking import append_experiment

LABEL_MAP = {"NOT_PLANET": 0, "PLANET": 1}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_test_split():
    df = load_labeled_tce("data/raw/q1_q17_dr25_tce_2026_07_08_22_53_14.csv",
                           koi_cache_path="data/koi_cumulative.csv")
    label_source = "unknown"
    if "label_source" in df.columns:
        vals = df["label_source"].dropna().unique()
        if len(vals):
            label_source = str(vals[0])

    X = build_feature_matrix(df)
    y = df["label"].map(LABEL_MAP).values
    groups = df["kepid"]
    _, _, test_idx = grouped_train_val_test_split(df, groups, "label")
    return X.iloc[test_idx], y[test_idx], label_source


def score_row(name, y_true, y_pred, y_prob):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
    }


def evaluate_ml_models(X_test, y_test):
    rows = []
    for fname in ["logistic_regression", "random_forest", "gradient_boosting", "svm"]:
        path = f"models/ml/{fname}.joblib"
        if not os.path.exists(path):
            continue
        pipe = joblib.load(path)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        rows.append(score_row(fname, y_test, y_pred, y_prob))
    return rows


def evaluate_nn_model(model_name, X_test, y_test):
    ckpt = f"models/dl/{model_name}_final.pt"
    if not os.path.exists(ckpt):
        return None
    pre = joblib.load("models/dl/preprocessor.joblib")
    feature_cols = joblib.load("models/dl/feature_columns.joblib")
    X_proc = pre.transform(X_test[feature_cols])

    config = joblib.load(f"models/dl/{model_name}_config.joblib")
    if model_name == "mlp":
        model = MLPClassifier(n_features=config["n_features"], n_classes=config["n_classes"])
    else:
        model = TabTransformer(n_features=config["n_features"], n_classes=config["n_classes"],
                                d_model=config["d_model"], n_heads=config["n_heads"], n_layers=config["n_layers"])
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.to(DEVICE).eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_proc, dtype=torch.float32).to(DEVICE))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()

    return score_row(model_name, y_test, preds, probs)


def _sequence_matrix_from_frame(df):
    rows = []
    labels = []
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
            arr = np.asarray([0.0], dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.size == 0:
            arr = np.zeros(64, dtype=np.float32)
        if arr.size < 64:
            arr = np.pad(arr, (0, 64 - arr.size), constant_values=0.0)
        if arr.size > 64:
            arr = arr[:64]
        rows.append(arr)
        labels.append(LABEL_MAP.get(str(row.get("label", 0)), 0))
    return np.vstack(rows).astype(np.float32), np.asarray(labels, dtype=np.int64)


def evaluate_sequence_models():
    """Score saved CNN1D and RNN checkpoints, if they exist.

    The project currently falls back to a synthetic light-curve-style sample when
    no real time-series archive is present. This is enough to keep the new
    experiment surface consistent and testable without pretending there is a
    full light-curve dataset in the sandbox.
    """
    rows = []
    frame, target = load_lightcurve_sequences(data_root="data/lightcurves", max_samples=5000)
    df = frame.assign(label=target)
    X, y = _sequence_matrix_from_frame(df)

    for model_name, model_cls, ckpt in [
        ("cnn1d", CNN1DClassifier, "models/sequence/cnn_final.pt"),
        ("rnn_gru", RNNClassifier, "models/sequence/rnn_final.pt"),
    ]:
        if not os.path.exists(ckpt):
            continue
        if model_name == "cnn1d":
            model = CNN1DClassifier(n_features=X.shape[1], n_classes=2).to(DEVICE)
        else:
            model = RNNClassifier(input_size=1, hidden_size=32, n_classes=2, n_layers=2).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.to(DEVICE).eval()

        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
        rows.append(score_row(model_name, y, preds, probs))
    return rows


def main():
    X_test, y_test, label_source = get_test_split()
    rows = evaluate_ml_models(X_test, y_test)
    for nn_name in ["mlp", "transformer"]:
        r = evaluate_nn_model(nn_name, X_test, y_test)
        if r:
            rows.append(r)

    rows.extend(evaluate_sequence_models())

    table = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    os.makedirs("reports", exist_ok=True)
    table.to_csv("reports/model_comparison.csv", index=False)
    with open("reports/label_source.txt", "w", encoding="utf-8") as handle:
        handle.write(str(label_source) + "\n")

    metrics_row = table.sort_values("f1", ascending=False).iloc[0].to_dict() if not table.empty else {}
    append_experiment(
        dataset_variant="KOI",
        model_type="ensemble_metrics",
        model_name="model_comparison",
        metrics=metrics_row,
        notes=f"label_source={label_source}",
    )

    print(table.to_string(index=False))
    print("\nSaved -> reports/model_comparison.csv")
    print(f"Saved label source -> reports/label_source.txt ({label_source})")
    return table


if __name__ == "__main__":
    main()
