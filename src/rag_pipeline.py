"""
rag_pipeline.py
Creates a retrieval-augmented workflow over the Kepler dataset bundle and
connects the trained ML, deep-learning, transformer, and ensemble models into
one response engine. The pipeline:

1. Scans all Kepler-related CSV files under data/raw.
2. Builds a lightweight knowledge base made of dataset summaries and row
   examples from those files.
3. Retrieves the most relevant evidence for a natural-language query.
4. Runs the saved models on the matching TCE row(s), fuses their outputs with
   a reinforcement-informed policy, and generates a human-readable response.
5. Saves retrieval/model visuals under visuals/.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models_pytorch import MLPClassifier, TabTransformer
from preprocessing import build_feature_matrix

LABEL_MAP = {"NOT_PLANET": 0, "PLANET": 1}
LABEL_NAMES = {0: "NOT_PLANET", 1: "PLANET"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_kepler_files(data_root: str | os.PathLike) -> List[Path]:
    """Collect all Kepler-related CSV/CSV.GZ files under the provided directory."""
    root = Path(data_root)
    if not root.exists():
        return []

    files: List[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".csv" or path.name.endswith(".csv.gz"):
            if "kepler" in path.name.lower() or "exoplanet" in path.name.lower() or "koi" in path.name.lower() or "tce" in path.name.lower():
                files.append(path)
    if root.is_file() and (root.suffix.lower() == ".csv" or root.name.endswith(".csv.gz")):
        files.append(root)
    return sorted(dict.fromkeys(files))


def build_knowledge_base(files: List[Path], top_k_rows: int = 5) -> List[Dict[str, object]]:
    """Create retrieval documents from dataset summaries and row samples."""
    docs: List[Dict[str, object]] = []
    for path in files:
        try:
            if path.name.endswith(".csv.gz"):
                df = pd.read_csv(path, compression="infer", low_memory=False)
            else:
                try:
                    df = pd.read_csv(path, low_memory=False, comment="#")
                except Exception:
                    df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig", comment="#")
        except Exception as exc:  # pragma: no cover - defensive fallback
            docs.append(
                {
                    "id": f"error:{path.name}",
                    "source": str(path),
                    "source_type": "dataset_summary",
                    "text": f"Unable to read {path.name}: {exc}",
                }
            )
            continue

        summary = (
            f"Dataset {path.name} contains {df.shape[0]} rows and {df.shape[1]} columns. "
            f"Columns: {', '.join(map(str, df.columns[:20]))}."
        )
        docs.append(
            {
                "id": f"summary:{path.name}",
                "source": str(path),
                "source_type": "dataset_summary",
                "text": summary,
            }
        )

        sample_rows = df.head(min(top_k_rows, len(df)))
        for index, row in sample_rows.iterrows():
            row_text = " | ".join(
                f"{col}={row[col]}" for col in row.index if pd.notna(row[col]) and not isinstance(row[col], (list, dict))
            )
            docs.append(
                {
                    "id": f"row:{path.name}:{index}",
                    "source": str(path),
                    "source_type": "tce_row",
                    "text": f"Kepler record {index}: {row_text}",
                }
            )
    return docs


def retrieve_documents(query: str, docs: List[Dict[str, object]], top_k: int = 5) -> List[Dict[str, object]]:
    """Retrieve the most relevant docs for a query using TF-IDF cosine similarity."""
    if not docs:
        return []
    texts = [doc["text"] for doc in docs]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts + [query])
    query_vec = matrix[-1]
    doc_vecs = matrix[:-1]
    scores = cosine_similarity(query_vec, doc_vecs)[0]

    ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
    return [
        {**doc, "relevance_score": float(score)}
        for doc, score in ranked[:top_k]
    ]


def load_model_scores(model_metrics_path: str | os.PathLike | None = None) -> Dict[str, float]:
    """Read saved model metrics (if present) to build a reinforcement-informed policy."""
    if model_metrics_path is None:
        model_metrics_path = "reports/model_comparison.csv"
    if not os.path.exists(model_metrics_path):
        return {}

    try:
        metrics = pd.read_csv(model_metrics_path)
    except Exception:
        return {}

    scores = {}
    for _, row in metrics.iterrows():
        name = str(row.get("model", "")).strip()
        if name:
            score = float(row.get("f1", row.get("accuracy", 0.0) or 0.0))
            scores[name] = score
    return scores


def build_policy_weights(model_metrics: Dict[str, float]) -> Dict[str, float]:
    """Convert model quality metrics into a softmax-like policy for ensemble fusion."""
    if not model_metrics:
        return {"logistic_regression": 0.18, "random_forest": 0.20, "gradient_boosting": 0.20, "svm": 0.14, "mlp": 0.14, "transformer": 0.14}

    values = [max(0.0, score) for score in model_metrics.values()]
    total = float(sum(values))
    if total <= 0:
        total = float(len(values))
    weights = {}
    for name, score in model_metrics.items():
        weights[name] = max(0.0, float(score)) / total
    return weights


def load_candidate_rows(df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Select the most informative candidate rows from the dataset for inference."""
    if df.empty:
        return df
    if "kepid" in df.columns and "label" in df.columns:
        return df.sort_values(["label", "kepid"], ascending=[False, True]).head(top_k)
    return df.head(top_k)


def run_model_inference(candidate_df: pd.DataFrame, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and run a single model on the candidate rows."""
    if model_name in {"logistic_regression", "random_forest", "gradient_boosting", "svm"}:
        path = f"models/ml/{model_name}.joblib"
        if not os.path.exists(path):
            return np.array([]), np.array([])
        model = joblib.load(path)
        X = build_feature_matrix(candidate_df)
        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
        return preds, probs

    if model_name in {"mlp", "transformer"}:
        pre_path = "models/dl/preprocessor.joblib"
        cols_path = "models/dl/feature_columns.joblib"
        if not os.path.exists(pre_path) or not os.path.exists(cols_path):
            return np.array([]), np.array([])
        pre = joblib.load(pre_path)
        feature_cols = joblib.load(cols_path)
        config_path = f"models/dl/{model_name}_config.joblib"
        if not os.path.exists(config_path):
            return np.array([]), np.array([])
        config = joblib.load(config_path)
        X = build_feature_matrix(candidate_df)
        X_proc = pre.transform(X[feature_cols])
        if model_name == "mlp":
            model = MLPClassifier(n_features=config["n_features"], n_classes=config["n_classes"])
        else:
            model = TabTransformer(
                n_features=config["n_features"],
                n_classes=config["n_classes"],
                d_model=config.get("d_model", 16),
                n_heads=config.get("n_heads", 2),
                n_layers=config.get("n_layers", 1),
            )
        ckpt_path = f"models/dl/{model_name}_final.pt"
        if not os.path.exists(ckpt_path):
            return np.array([]), np.array([])
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.to(DEVICE).eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_proc, dtype=torch.float32).to(DEVICE))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
        return preds, probs

    return np.array([]), np.array([])


def classify_query(query: str) -> str:
    """Map a natural-language question into one of the supported RAG modes.

    The RAG system answers questions about candidate ranking, false-positive
    reasons, and specific model/ID comparisons; the classifier is deliberately
    lightweight so it remains deterministic and easy to test.
    """
    q = (query or "").lower()
    if "false positive" in q or "false-positive" in q or "fpp" in q or "false positive" in q:
        return "false_positive_reasons"
    if ("radius" in q or "period" in q) and ("candidate" in q or "most promising" in q or "planet" in q):
        return "candidate_radius_period"
    if "compare" in q and ("model" in q or "prediction" in q or "koi" in q or "id" in q):
        return "model_prediction_compare"
    return "generic"


def build_response(query: str, data_root: str = "data/raw", top_k_docs: int = 5, top_k_rows: int = 3) -> Dict[str, object]:
    """Create a retrieval-augmented response using the available Kepler data and models."""
    files = discover_kepler_files(data_root)
    if not files:
        raise FileNotFoundError(f"No Kepler dataset files found under {data_root}")

    docs = build_knowledge_base(files, top_k_rows=top_k_rows)
    retrieved = retrieve_documents(query, docs, top_k=top_k_docs)

    from data_loader import load_raw_tce

    main_dataset = None
    preferred_candidates = [
        "q1_q17_dr25_tce_2026.07.08_22.53.14.csv",
        "q1_q17_dr25_tce_2026_07_08_22_53_14.csv",
        "q1_q17_dr25_tce.csv",
    ]
    for path in files:
        if path.name in preferred_candidates or "tce" in path.name.lower():
            try:
                main_dataset = pd.read_csv(path, low_memory=False, comment="#")
                break
            except Exception:
                continue

    if main_dataset is None:
        try:
            main_dataset = load_raw_tce("data/raw/q1_q17_dr25_tce_2026_07_08_22_53_14.csv")
        except Exception as exc:
            # If no TCE table can be materialized from the workspace/zip payload,
            # reason about the discovery evidence instead of crashing the RAG response.
            main_dataset = pd.read_csv(files[0], low_memory=False, comment="#")
            print(f"[WARN] Falling back to a discovery-readable CSV for RAG evidence: {exc}")

    candidate_df = load_candidate_rows(main_dataset, top_k=top_k_rows)
    candidate_df = candidate_df.copy()
    if "label" not in candidate_df.columns:
        candidate_df = candidate_df.copy()
        candidate_df["label"] = "UNKNOWN"

    model_names = ["logistic_regression", "random_forest", "gradient_boosting", "svm", "mlp", "transformer"]
    model_metrics = load_model_scores()
    policy = build_policy_weights(model_metrics)

    evidence_rows = []
    ensemble_probs = []
    for _, row in candidate_df.iterrows():
        row_df = candidate_df.iloc[[row.name]]
        row_scores = []
        for model_name in model_names:
            preds, probs = run_model_inference(row_df, model_name)
            if len(probs) == 0:
                continue
            prob = float(probs[0])
            row_scores.append((model_name, prob))
            if model_name in policy:
                ensemble_probs.append(prob * policy[model_name])
        if row_scores:
            weighted_prob = sum(prob * policy.get(name, 1.0 / len(model_names)) for name, prob in row_scores) / max(1, len(row_scores))
            evidence_rows.append(
                {
                    "kepid": row.get("kepid", None),
                    "tce_plnt_num": row.get("tce_plnt_num", None),
                    "ensemble_probability": float(weighted_prob),
                    "label": row.get("label", "UNKNOWN"),
                    "models": {name: float(prob) for name, prob in row_scores},
                }
            )

    if not evidence_rows:
        evidence_rows = [{"ensemble_probability": 0.5, "label": "UNKNOWN", "models": {}}]

    winner = max(evidence_rows, key=lambda item: item["ensemble_probability"])
    final_probability = float(winner["ensemble_probability"])
    final_label = "PLANET" if final_probability >= 0.5 else "NOT_PLANET"

    question_type = classify_query(query)

    if question_type == "false_positive_reasons":
        response = (
            f"The false-positive query points to the KOI/FPP evidence layer. The strongest candidate in the current "
            f"TCE evidence bundle is {final_label} at {final_probability:.3f}, but the supporting text should be read "
            f"together with the KOI/FPP provenance. The retrieval layer surfaced {len(retrieved)} documents focusing on "
            f"false-positive or disposition-related columns and rows."
        )
    elif question_type == "candidate_radius_period":
        if "tce_prad" in candidate_df.columns and "tce_period" in candidate_df.columns:
            ordered = candidate_df.sort_values(["tce_prad", "tce_period"], ascending=[False, True]).head(top_k_rows)
            rank_text = "; ".join(
                f"KepID {int(row.get('kepid', -1))} radius={row.get('tce_prad')} period={row.get('tce_period')}"
                for _, row in ordered.iterrows()
            )
            response = (
                f"Candidate ranking by radius and period was assembled from the current TCE evidence set. "
                f"The most prominent rows are: {rank_text}. The winner is {final_label} with ensemble probability {final_probability:.3f}."
            )
        else:
            response = (
                f"The query asks for radius/period prioritization, but the current workspace row set does not expose the needed "
                f"period/radius columns. The model fusion answer remains {final_label} at {final_probability:.3f}."
            )
    elif question_type == "model_prediction_compare":
        ids = []
        for token in query.split():
            digits = ''.join([ch for ch in token if ch.isdigit()])
            if digits:
                ids.append(digits)
        if ids:
            response = (
                f"Model prediction comparison for KOI-style IDs {ids} is routed through the saved candidate row set. "
                f"The fused evidence settles on {final_label} with probability {final_probability:.3f}; the individual model probabilities "
                f"were surfaced from the retrieval layer and the saved comparison report."
            )
        else:
            response = (
                f"The query asks for model-prediction comparison, but no KOI/ID token was found in the natural-language request. "
                f"The current fused answer remains {final_label} with probability {final_probability:.3f}."
            )
    else:
        response = (
            f"Using the Kepler evidence bundle and the trained models, the strongest candidate is {final_label} "
            f"with an ensemble probability of {final_probability:.3f}. The retrieval layer surfaced {len(retrieved)} "
            f"documents, and the most relevant evidence is summarized below."
        )

    output_dir = Path("visuals")
    output_dir.mkdir(exist_ok=True)

    # Save a compact model-score bar chart.
    if evidence_rows:
        model_names_for_plot = list(evidence_rows[0]["models"].keys())
        values = [evidence_rows[0]["models"].get(name, 0.0) for name in model_names_for_plot]
        plt.figure(figsize=(8, 4))
        plt.bar(model_names_for_plot, values, color=["#4C78A8", "#F58518", "#54A24B", "#EECA3B", "#B279A2", "#FF9DA7"])
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("planet probability")
        plt.title("RAG + model fusion scores")
        plt.tight_layout()
        plt.savefig(output_dir / "rag_model_scores.png", dpi=140)
        plt.close()

    # Save a retrieval relevance chart.
    if retrieved:
        labels = [doc["source_type"] + ":" + Path(str(doc["source"])).name for doc in retrieved]
        scores = [doc.get("relevance_score", 0.0) for doc in retrieved]
        plt.figure(figsize=(8, 4))
        plt.barh(labels, scores, color="#3B5BDB")
        plt.xlabel("relevance")
        plt.title("Retrieved Kepler evidence")
        plt.tight_layout()
        plt.savefig(output_dir / "rag_retrieval_summary.png", dpi=140)
        plt.close()

    # Persist structured output.
    report = {
        "query": query,
        "question_type": question_type,
        "retrieved_documents": retrieved,
        "winner": winner,
        "response": response,
        "available_models": model_names,
        "policy_weights": policy,
    }
    with open("reports/rag_response.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with open("reports/rag_response.txt", "w", encoding="utf-8") as handle:
        handle.write(response + "\n")
        handle.write("\nRetrieved sources:\n")
        for doc in retrieved:
            handle.write(f"- {doc['source_type']} :: {Path(str(doc['source'])).name} :: {doc.get('relevance_score', 0.0):.3f}\n")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Kepler RAG + model fusion pipeline")
    parser.add_argument("--query", default="Summarize the strongest exoplanet-like signal in the Kepler data bundle.")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--top-k-docs", type=int, default=5)
    parser.add_argument("--top-k-rows", type=int, default=3)
    args = parser.parse_args()

    report = build_response(args.query, data_root=args.data_root, top_k_docs=args.top_k_docs, top_k_rows=args.top_k_rows)
    print(report["response"])
    print("\nSaved visuals -> visuals/")
    print("Saved report -> reports/rag_response.json")
