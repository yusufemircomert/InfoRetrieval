"""Evaluation metrics for multi-label toxic comment classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import LABELS


def compute_per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    label_names = labels or LABELS
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    rows = []
    for idx, label in enumerate(label_names):
        rows.append(
            {
                "label": label,
                "precision": precision_score(y_true[:, idx], y_pred[:, idx], zero_division=0),
                "recall": recall_score(y_true[:, idx], y_pred[:, idx], zero_division=0),
                "f1": f1_score(y_true[:, idx], y_pred[:, idx], zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def compute_macro_micro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    per_label = compute_per_label_metrics(y_true, y_pred, labels=labels)
    summary = compute_macro_micro_f1(y_true, y_pred)
    return {
        "per_label": per_label,
        "macro_f1": summary["macro_f1"],
        "micro_f1": summary["micro_f1"],
    }


def summarize_across_folds(fold_results: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate per-label and summary metrics across folds (mean ± std)."""
    label_frames = [result["per_label"].set_index("label") for result in fold_results]
    combined = pd.concat(label_frames, keys=range(len(label_frames)), names=["fold"])

    per_label_summary = combined.groupby("label").agg(["mean", "std"]).round(4)
    per_label_summary.columns = [f"{metric}_{stat}" for metric, stat in per_label_summary.columns]

    summary_rows = []
    for metric in ("macro_f1", "micro_f1"):
        values = [result[metric] for result in fold_results]
        summary_rows.append(
            {
                "label": metric,
                "precision_mean": np.nan,
                "precision_std": np.nan,
                "recall_mean": np.nan,
                "recall_std": np.nan,
                "f1_mean": float(np.mean(values)),
                "f1_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        )

    summary_df = pd.DataFrame(summary_rows).set_index("label")
    return pd.concat([per_label_summary, summary_df])
