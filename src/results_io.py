"""Save and load experiment metrics to CSV."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import LABELS, RESULTS_DIR


def _metrics_to_rows(
    fold_idx: int,
    method: str,
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in metrics["per_label"].iterrows():
        rows.append(
            {
                "fold": fold_idx,
                "method": method,
                "label": row["label"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
            }
        )
    return rows


def save_fold_metrics(
    fold_idx: int,
    method: str,
    metrics: dict[str, Any],
    output_dir: Path | None = None,
) -> Path:
    out_dir = output_dir or (RESULTS_DIR / method)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _metrics_to_rows(fold_idx, method, metrics)
    out_path = out_dir / f"fold_{fold_idx}_metrics.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def save_all_fold_metrics(
    fold_results: list[dict[str, Any]],
    method: str,
    output_dir: Path | None = None,
) -> Path:
    out_dir = output_dir or (RESULTS_DIR / method)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for result in fold_results:
        all_rows.extend(
            _metrics_to_rows(result["fold"], method, result["metrics"])
        )

    out_path = out_dir / "all_folds_metrics.csv"
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    return out_path


def save_summary_json(
    fold_results: list[dict[str, Any]],
    method: str,
    output_dir: Path | None = None,
) -> Path:
    out_dir = output_dir or (RESULTS_DIR / method)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for result in fold_results:
        summary.append(
            {
                "fold": result["fold"],
                "macro_f1": result["metrics"]["macro_f1"],
                "micro_f1": result["metrics"]["micro_f1"],
            }
        )

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path
