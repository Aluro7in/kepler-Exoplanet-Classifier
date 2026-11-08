"""Simple experiment index for the Exoplanet AI Lab.

The project already saves model-comparison CSVs in reports/. This module adds
an append-only experiment registry so each training/evaluation cycle can be
indexed without overwriting old evidence. Smaller experiments can add one row
per model or one row per experiment family.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

INDEX_PATH = Path("reports/experiment_index.csv")

DEFAULT_COLUMNS = [
    "dataset_variant",
    "model_type",
    "model_name",
    "f1",
    "accuracy",
    "roc_auc",
    "run_timestamp",
    "notes",
]


def _ensure_index_file(index_path: str | os.PathLike = INDEX_PATH) -> Path:
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=DEFAULT_COLUMNS)
            writer.writeheader()
    return path


def append_experiment(
    dataset_variant: str,
    model_type: str,
    model_name: str,
    metrics: dict[str, float] | None = None,
    run_timestamp: str | None = None,
    notes: str = "",
    index_path: str | os.PathLike = INDEX_PATH,
) -> dict[str, Any]:
    """Append one experiment record to the index file.

    The function is deliberately small and idempotent. It appends one row and
    returns the row payload to allow caller-side unit tests and CLI workflows to
    inspect what was saved.
    """
    path = _ensure_index_file(index_path)
    if run_timestamp is None:
        run_timestamp = datetime.now(timezone.utc).isoformat()
    metrics = metrics or {}

    row = {
        "dataset_variant": dataset_variant,
        "model_type": model_type,
        "model_name": model_name,
        "f1": metrics.get("f1", 0.0),
        "accuracy": metrics.get("accuracy", 0.0),
        "roc_auc": metrics.get("roc_auc", 0.0),
        "run_timestamp": run_timestamp,
        "notes": notes,
    }

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_COLUMNS)
        writer.writerow(row)

    return row


def read_experiment_index(index_path: str | os.PathLike = INDEX_PATH) -> list[dict[str, str]]:
    """Read the CSV experiment index into memory for lightweight reporting."""
    path = Path(index_path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)
