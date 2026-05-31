"""Fine-tuned DistilBERT for multi-label toxic comment classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.config import (
    BERT_BATCH_SIZE,
    BERT_EPOCHS,
    BERT_LEARNING_RATE,
    BERT_MODEL_DIR,
    BERT_MODEL_NAME,
    LABELS,
    MAX_SEQ_LENGTH,
    RANDOM_SEED,
)
from src.metrics import evaluate_predictions


def _batch_tokenize(
    tokenizer,
    texts: list[str],
    max_length: int,
    chunk_size: int = 2000,
) -> dict[str, list]:
    """Tokenize in chunks so progress is visible and RAM stays bounded."""
    merged: dict[str, list] = {}
    total = len(texts)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        print(f"    tokenize {start}-{end} / {total}...", flush=True)
        chunk = tokenizer(
            texts[start:end],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        for key, values in chunk.items():
            merged.setdefault(key, []).extend(values)
    return merged


class ToxicCommentDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray | None,
        tokenizer,
        max_length: int = MAX_SEQ_LENGTH,
    ) -> None:
        self.labels = labels
        self.encodings = _batch_tokenize(tokenizer, texts, max_length)

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


@dataclass
class BertFoldResult:
    fold_idx: int
    metrics: dict[str, Any]
    y_true: np.ndarray
    y_pred: np.ndarray


def _resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_local_only() -> bool:
    return BERT_MODEL_DIR.is_dir() and (BERT_MODEL_DIR / "config.json").exists()


def _build_trainer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str,
    device: torch.device,
) -> tuple[Trainer, AutoTokenizer]:
    local_only = _load_local_only()
    print(f"  Loading tokenizer from {'local' if local_only else 'hub'}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, local_files_only=local_only)
    print(f"  Loading model from {'local' if local_only else 'hub'}...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        local_files_only=local_only,
    )
    print("  Model loaded.", flush=True)

    print(f"  Tokenizing train ({len(train_df)})...", flush=True)
    train_dataset = ToxicCommentDataset(
        train_df["comment_text"].tolist(),
        train_df[LABELS].to_numpy(),
        tokenizer,
    )
    print(f"  Tokenizing val ({len(val_df)})...", flush=True)
    val_dataset = ToxicCommentDataset(
        val_df["comment_text"].tolist(),
        val_df[LABELS].to_numpy(),
        tokenizer,
    )

    use_fp16 = device.type == "cuda"
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=BERT_EPOCHS,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE,
        learning_rate=BERT_LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        logging_first_step=True,
        seed=RANDOM_SEED,
        report_to="none",
        use_cpu=(device.type == "cpu"),
        fp16=use_fp16,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    return trainer, tokenizer


def predict_dataframe(
    model,
    tokenizer,
    df: pd.DataFrame,
    device: torch.device,
    batch_size: int = BERT_BATCH_SIZE,
) -> np.ndarray:
    model.to(device)
    model.eval()

    dataset = ToxicCommentDataset(df["comment_text"].tolist(), None, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            all_probs.append(probs)

    probs = np.vstack(all_probs)
    return (probs >= 0.5).astype(int)


def train_and_evaluate_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
    device: str | None = None,
) -> BertFoldResult:
    resolved_device = _resolve_device(device)
    trainer, tokenizer = _build_trainer(train_df, val_df, output_dir, resolved_device)
    print("  Training...", flush=True)
    trainer.train()

    print(f"  Predicting on test ({len(test_df)})...", flush=True)
    y_true = test_df[LABELS].to_numpy()
    y_pred = predict_dataframe(trainer.model, tokenizer, test_df, resolved_device)
    metrics = evaluate_predictions(y_true, y_pred)

    return BertFoldResult(
        fold_idx=fold_idx,
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
    )
