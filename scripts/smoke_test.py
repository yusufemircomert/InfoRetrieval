import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_utils import prepare_dataset
from src.llm_prompts import build_zero_shot_prompt, parse_llm_output

df, folds = prepare_dataset(force_recreate_subsample=True, force_recreate_folds=True)
print("Subsample size:", len(df))
print("Fold 0 test size:", len(folds[0]["test"]))
print("Parse test:", parse_llm_output('{"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0}'))
print("Prompt chars:", len(build_zero_shot_prompt("hello world")))
