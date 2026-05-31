"""Run LLM zero-shot / few-shot CV from the command line (GPU)."""

from __future__ import annotations

import argparse
import gc
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LLM_MODEL_NAME, N_FOLDS, RESULTS_DIR
from src.data_utils import get_fold_dataframes, prepare_dataset
from src.llm_inference import LocalLLMClassifier, clear_gpu_memory, evaluate_llm_fold
from src.metrics import summarize_across_folds
from src.results_io import save_all_fold_metrics, save_fold_metrics, save_summary_json

MODES = ("zero_shot", "few_shot")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def verify_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Activate .venv and use the fuzzy-final kernel."
        )
    try:
        torch.randn(1, device="cuda")
    except RuntimeError as exc:
        raise RuntimeError("GPU detected but CUDA kernels failed.") from exc
    name = torch.cuda.get_device_name(0)
    log(f"Using GPU: {name}")
    return "cuda"


def fold_mode_done(fold_idx: int, mode: str) -> bool:
    return (RESULTS_DIR / "llm" / mode / f"fold_{fold_idx}_metrics.csv").exists()


def rebuild_consolidated_metrics(mode: str) -> None:
    """Merge all per-fold CSVs (including fold 0 from earlier runs) into one file."""
    mode_dir = RESULTS_DIR / "llm" / mode
    if not mode_dir.exists():
        return

    fold_files = sorted(mode_dir.glob("fold_*_metrics.csv"))
    if not fold_files:
        return

    frames = [pd.read_csv(path) for path in fold_files]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(mode_dir / "all_folds_metrics.csv", index=False)

    summary = (
        combined.groupby("fold")[["macro_f1", "micro_f1"]]
        .first()
        .reset_index()
        .sort_values("fold")
    )
    summary.to_json(mode_dir / "summary.json", orient="records", indent=2)
    log(f"Updated {mode_dir / 'all_folds_metrics.csv'} ({len(summary)} folds)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM cross-validation")
    parser.add_argument(
        "--folds",
        type=str,
        default="1,2,3,4",
        help="Fold indices comma-separated or 'all' (default: 1,2,3,4)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="zero_shot,few_shot",
        help="Prompt modes comma-separated (default: zero_shot,few_shot)",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="Skip fold/mode pairs that already have metrics CSV",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Use cpu if you need GPU for other apps (slower but no OOM)",
    )
    args = parser.parse_args()

    device = args.device
    if device == "cuda":
        verify_cuda()
    else:
        log("Using CPU (slower, but avoids GPU memory conflicts)")

    if args.folds == "all":
        folds_to_run = list(range(N_FOLDS))
    else:
        folds_to_run = [int(x.strip()) for x in args.folds.split(",")]

    modes_to_run = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes_to_run:
        if mode not in MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {MODES}")

    df, folds = prepare_dataset()
    fold_dfs = get_fold_dataframes(df, folds)

    log(f"LLM model: {LLM_MODEL_NAME}")
    log(f"Folds: {folds_to_run} | Modes: {modes_to_run}")

    session_results: dict[str, list[dict]] = {mode: [] for mode in modes_to_run}

    for fold_idx in folds_to_run:
        test_df = fold_dfs[fold_idx]["test"]
        log(f"=== Fold {fold_idx} | {len(test_df)} test comments ===")

        for mode in modes_to_run:
            if args.skip_done and fold_mode_done(fold_idx, mode):
                log(f"  Skipping {mode} (already done)")
                continue

            log(f"  START {mode}")
            clear_gpu_memory()
            classifier = LocalLLMClassifier(device=device)
            try:
                result = evaluate_llm_fold(
                    classifier=classifier,
                    fold_idx=fold_idx,
                    test_df=test_df,
                    mode=mode,  # type: ignore[arg-type]
                )
            finally:
                classifier.unload()
                del classifier
                gc.collect()
                clear_gpu_memory()

            save_fold_metrics(
                fold_idx,
                f"llm_{mode}",
                result.metrics,
                output_dir=RESULTS_DIR / "llm" / mode,
            )
            session_results[mode].append({"fold": fold_idx, "metrics": result.metrics})
            log(
                f"  DONE {mode} | macro F1: {result.metrics['macro_f1']:.4f} "
                f"| micro F1: {result.metrics['micro_f1']:.4f}"
            )

    for mode in modes_to_run:
        rebuild_consolidated_metrics(mode)
        if session_results[mode]:
            log(f"\n--- {mode} (this session) ---")
            print(
                summarize_across_folds([r["metrics"] for r in session_results[mode]]),
                flush=True,
            )

    log(f"Finished. Results in {RESULTS_DIR / 'llm'}")


if __name__ == "__main__":
    main()
