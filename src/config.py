"""Shared configuration for the final project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FOLD_DIR = DATA_DIR / "fold_indices"

TRAIN_CSV = PROJECT_ROOT / "train.csv"
SUBSAMPLED_CSV = DATA_DIR / "train_subsampled.csv"

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

RANDOM_SEED = 42
N_FOLDS = 5
SUBSAMPLE_SIZE = 25_000
MAX_SEQ_LENGTH = 256

# BERT training defaults (tuned for local GPU; reduce batch size if OOM)
BERT_MODEL_DIR = PROJECT_ROOT / "models" / "distilbert-base-uncased"
BERT_MODEL_NAME = str(BERT_MODEL_DIR)  # local copy — avoids HF download hangs
BERT_EPOCHS = 3
BERT_BATCH_SIZE = 16
BERT_LEARNING_RATE = 2e-5

# Local LLM via Hugging Face (no Ollama required)
LLM_MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
LLM_MAX_NEW_TOKENS = 64  # JSON output is short; lower = less VRAM
LLM_TEMPERATURE = 0.0
