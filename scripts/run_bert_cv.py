"""Run BERT 5-fold cross-validation from the command line (GPU)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bert_model import train_and_evaluate_fold
from src.config import N_FOLDS, RESULTS_DIR
from src.data_utils import get_fold_dataframes, prepare_dataset
from src.metrics import summarize_across_folds
from src.results_io import save_all_fold_metrics, save_fold_metrics, save_summary_json


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def verify_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Activate .venv and install PyTorch cu128:\n"
            "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
        )
    try:
        torch.randn(1, device="cuda")
    except RuntimeError as exc:
        raise RuntimeError(
            "GPU detected but CUDA kernels failed. RTX 5060 needs PyTorch cu128."
        ) from exc
    name = torch.cuda.get_device_name(0)
    log(f"Using GPU: {name}")
    return "cuda"


def fold_already_done(fold_idx: int) -> bool:
    return (RESULTS_DIR / "bert" / f"fold_{fold_idx}_metrics.csv").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BERT cross-validation")
    parser.add_argument(
        "--folds",
        type=str,
        default="all",
        help="Fold indices comma-separated (e.g. 0,1) or 'all'",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="Skip folds that already have metrics CSV",
    )
    args = parser.parse_args()

    device = verify_cuda()

    if args.folds == "all":
        folds_to_run = list(range(N_FOLDS))
    else:
        folds_to_run = [int(x.strip()) for x in args.folds.split(",")]

    df, folds = prepare_dataset()
    fold_dfs = get_fold_dataframes(df, folds)

    checkpoint_root = RESULTS_DIR / "bert" / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "bert").mkdir(parents=True, exist_ok=True)

    bert_results = []
    for fold_idx in folds_to_run:
        if args.skip_done and fold_already_done(fold_idx):
            log(f"Skipping fold {fold_idx} (already done)")
            continue

        fold = fold_dfs[fold_idx]
        output_dir = str(checkpoint_root / f"fold_{fold_idx}")
        log(f"=== Fold {fold_idx} START ===")
        result = train_and_evaluate_fold(
            fold_idx=fold_idx,
            train_df=fold["train"],
            val_df=fold["val"],
            test_df=fold["test"],
            output_dir=output_dir,
            device=device,
        )
        save_fold_metrics(fold_idx, "bert", result.metrics, output_dir=RESULTS_DIR / "bert")
        bert_results.append({"fold": fold_idx, "metrics": result.metrics})
        log(f"Fold {fold_idx} DONE | macro F1: {result.metrics['macro_f1']:.4f} | micro F1: {result.metrics['micro_f1']:.4f}")

    if bert_results:
        save_all_fold_metrics(bert_results, method="bert", output_dir=RESULTS_DIR / "bert")
        save_summary_json(bert_results, method="bert", output_dir=RESULTS_DIR / "bert")
        log("Summary across folds:")
        print(summarize_across_folds([r["metrics"] for r in bert_results]), flush=True)

    log(f"Finished. Results in {RESULTS_DIR / 'bert'}")


if __name__ == "__main__":
    main()
