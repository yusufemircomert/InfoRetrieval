"""Data loading, cleaning, subsampling, and cross-validation splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR,
    FOLD_DIR,
    LABELS,
    N_FOLDS,
    RANDOM_SEED,
    SUBSAMPLE_SIZE,
    SUBSAMPLED_CSV,
    TRAIN_CSV,
)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data(csv_path: Path | str | None = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path is not None else TRAIN_CSV
    df = pd.read_csv(path)
    expected = {"id", "comment_text", *LABELS}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    return " ".join(text.split())


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["comment_text"] = out["comment_text"].map(clean_text)
    for label in LABELS:
        out[label] = out[label].astype(int)
    return out


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in LABELS:
        count = int(df[label].sum())
        rows.append(
            {
                "label": label,
                "count": count,
                "percentage": 100.0 * count / len(df),
            }
        )
    return pd.DataFrame(rows)


def subsample_preserving_labels(
    df: pd.DataFrame,
    n_samples: int = SUBSAMPLE_SIZE,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    if len(df) <= n_samples:
        return df.reset_index(drop=True)

    x = df.index.to_numpy()
    y = df[LABELS].to_numpy()

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=n_samples,
        random_state=random_state,
    )
    _, selected_idx = next(splitter.split(x, y))
    sampled = df.iloc[selected_idx].reset_index(drop=True)
    return sampled


def create_or_load_subsample(
    force_recreate: bool = False,
    n_samples: int = SUBSAMPLE_SIZE,
) -> pd.DataFrame:
    ensure_dirs()
    if SUBSAMPLED_CSV.exists() and not force_recreate:
        return pd.read_csv(SUBSAMPLED_CSV)

    raw = load_raw_data()
    processed = preprocess_dataframe(raw)
    sampled = subsample_preserving_labels(processed, n_samples=n_samples)
    sampled.to_csv(SUBSAMPLED_CSV, index=False)
    return sampled


def build_cv_folds(
    df: pd.DataFrame,
    n_splits: int = N_FOLDS,
    random_state: int = RANDOM_SEED,
) -> list[dict[str, np.ndarray]]:
    x = np.arange(len(df))
    y = df[LABELS].to_numpy()

    mskf = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    folds: list[dict[str, np.ndarray]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(mskf.split(x, y)):
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=0.1,
            random_state=random_state + fold_idx,
            stratify=y[train_idx].sum(axis=1),
        )
        folds.append(
            {
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
            }
        )
    return folds


def save_folds(folds: list[dict[str, np.ndarray]], output_dir: Path | None = None) -> None:
    ensure_dirs()
    out_dir = output_dir or FOLD_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "n_folds": len(folds),
        "random_seed": RANDOM_SEED,
        "labels": LABELS,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for fold_idx, fold in enumerate(folds):
        fold_dir = out_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for split_name, indices in fold.items():
            np.save(fold_dir / f"{split_name}.npy", indices)


def load_folds(fold_dir: Path | None = None) -> list[dict[str, np.ndarray]]:
    base = fold_dir or FOLD_DIR
    if not base.exists():
        raise FileNotFoundError(f"Fold directory not found: {base}")

    folds: list[dict[str, np.ndarray]] = []
    fold_idx = 0
    while True:
        current = base / f"fold_{fold_idx}"
        if not current.exists():
            break
        folds.append(
            {
                "train": np.load(current / "train.npy"),
                "val": np.load(current / "val.npy"),
                "test": np.load(current / "test.npy"),
            }
        )
        fold_idx += 1

    if not folds:
        raise FileNotFoundError(f"No fold files found under {base}")
    return folds


def get_fold_dataframes(
    df: pd.DataFrame,
    folds: list[dict[str, np.ndarray]] | None = None,
) -> list[dict[str, pd.DataFrame]]:
    fold_indices = folds or load_folds()
    result: list[dict[str, pd.DataFrame]] = []
    for fold in fold_indices:
        result.append(
            {
                split: df.iloc[indices].reset_index(drop=True)
                for split, indices in fold.items()
            }
        )
    return result


def prepare_dataset(
    force_recreate_subsample: bool = False,
    force_recreate_folds: bool = False,
    n_samples: int = SUBSAMPLE_SIZE,
) -> tuple[pd.DataFrame, list[dict[str, np.ndarray]]]:
    ensure_dirs()
    df = create_or_load_subsample(force_recreate=force_recreate_subsample, n_samples=n_samples)

    if force_recreate_folds or not any(FOLD_DIR.glob("fold_0/*.npy")):
        folds = build_cv_folds(df)
        save_folds(folds)
    else:
        folds = load_folds()

    return df, folds
