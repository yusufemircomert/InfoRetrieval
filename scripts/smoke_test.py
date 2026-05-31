"""Quick sanity check for data loading, folds, and prompt parsing."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import N_FOLDS
from src.data_utils import prepare_dataset
from src.llm_prompts import build_zero_shot_prompt, parse_llm_output


def main() -> None:
    df, folds = prepare_dataset()
    assert len(df) > 0, "Subsample is empty"
    assert len(folds) == N_FOLDS, f"Expected {N_FOLDS} folds, got {len(folds)}"

    sample = (
        '{"toxic": 1, "severe_toxic": 0, "obscene": 0, '
        '"threat": 0, "insult": 1, "identity_hate": 0}'
    )
    parsed = parse_llm_output(sample)

    print("Subsample size:", len(df))
    print("Fold 0 test size:", len(folds[0]["test"]))
    print("Parse test:", parsed)
    print("Prompt chars:", len(build_zero_shot_prompt("hello world")))
    print("OK")


if __name__ == "__main__":
    main()
