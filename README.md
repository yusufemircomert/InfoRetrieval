# Toxic Comment Classification: Fine-Tuned BERT vs. Local LLM Prompting

**Course:** AIN 428 — Information Retrieval (final project)  
**Dataset:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)  
**Hardware tested:** NVIDIA GeForce RTX 5060 (8 GB VRAM, CUDA 12.8)

Multi-label toxic comment classification comparing **supervised DistilBERT fine-tuning** against **local LLM prompting** (zero-shot and few-shot), evaluated with **5-fold cross-validation** on identical stratified splits.

---

## Results at a glance

| Method | Macro F1 (mean ± std) | Micro F1 (mean ± std) |
|--------|----------------------:|----------------------:|
| **DistilBERT** | **0.424 ± 0.030** | **0.763 ± 0.006** |
| LLM zero-shot | 0.141 ± 0.001 | 0.156 ± 0.001 |
| LLM few-shot | 0.175 ± 0.002 | 0.171 ± 0.001 |

Fine-tuned DistilBERT clearly outperforms the local 1.7B LLM. The LLM achieves high recall but very low precision (over-predicts toxic labels). Both methods struggle on rare classes (`threat`, `identity_hate`).

Full tables and plots: `results/comparison/` or `notebooks/04_comparison.ipynb`.

---

## Quick start

### 1. Environment

```powershell
cd path\to\InfoRetrieval

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python -m ipykernel install --user --name fuzzy-final --display-name "Python 3.12 (fuzzy-final)"
```

### 2. Data and models (not in Git)

| Item | Location | How to get it |
|------|----------|---------------|
| Jigsaw training set | `train.csv` | [Kaggle download](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) |
| DistilBERT weights | `models/distilbert-base-uncased/` | `huggingface-cli download distilbert-base-uncased --local-dir models/distilbert-base-uncased` |
| Subsample | `data/train_subsampled.csv` | Created automatically from `train.csv` on first run |

Fold indices under `data/fold_indices/` are committed so BERT and LLM use the same splits. The LLM (`HuggingFaceTB/SmolLM2-1.7B-Instruct`) downloads from Hugging Face on first inference.

### 3. Run pipeline

```powershell
python scripts/smoke_test.py

python scripts/run_bert_cv.py --folds all --skip-done
python scripts/run_llm_cv.py --folds all --modes zero_shot,few_shot --skip-done

jupyter notebook notebooks/04_comparison.ipynb
```

Overnight wrappers (optional): `scripts/run_overnight.ps1`, `scripts/run_llm_overnight.ps1`.

---

## Project structure

```
.
├── Midterm-Final-Details.pdf     # Course assignment brief
├── requirements.txt
├── .gitignore
├── .gitattributes
├── train.csv                     # Not in Git — download locally
├── data/
│   ├── train_subsampled.csv      # Not in Git — auto-generated
│   └── fold_indices/             # Shared 5-fold splits (.npy + metadata.json)
├── models/
│   └── distilbert-base-uncased/  # Not in Git — download locally
├── src/
│   ├── config.py                 # Paths and hyperparameters
│   ├── data_utils.py             # Loading, subsampling, CV splits
│   ├── bert_model.py             # DistilBERT training and evaluation
│   ├── llm_inference.py          # Local LLM classifier with caching
│   ├── llm_prompts.py            # Prompt templates and JSON parsing
│   ├── metrics.py                # Precision, recall, F1
│   └── results_io.py             # Metrics export to CSV/JSON
├── notebooks/
│   ├── 01_eda_and_splits.ipynb   # EDA and fold creation
│   ├── 02_bert_cv.ipynb          # Interactive BERT CV
│   ├── 03_llm_cv.ipynb           # Interactive LLM CV
│   └── 04_comparison.ipynb       # Cross-method comparison and plots
├── scripts/
│   ├── run_bert_cv.py            # CLI: BERT 5-fold CV
│   ├── run_llm_cv.py             # CLI: LLM zero/few-shot CV
│   ├── smoke_test.py             # Pipeline sanity check
│   ├── run_overnight.ps1         # Long-running BERT wrapper
│   └── run_llm_overnight.ps1     # Long-running LLM wrapper
└── results/
    ├── bert/                     # Metrics CSVs + summary.json
    ├── llm/
    │   ├── zero_shot/
    │   ├── few_shot/
    │   └── cache/                # Not in Git — regeneratable
    └── comparison/               # Summary CSVs and PNG plots
```

---

## Methods

### DistilBERT

| Setting | Value |
|---------|-------|
| Model | `distilbert-base-uncased` (local copy) |
| Max length | 256 tokens |
| Epochs / batch / LR | 3 / 16 / 2e-5 |
| Threshold | 0.5 on sigmoid outputs |

### Local LLM

| Setting | Value |
|---------|-------|
| Model | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| Temperature | 0.0 |
| Max new tokens | 64 |
| Modes | Zero-shot and few-shot (3 in-context examples) |

Responses are parsed as JSON with regex fallback. Predictions are cached per fold/mode under `results/llm/cache/`.

---

## Results (5-fold CV, complete)

### Overall

| Fold | BERT macro | BERT micro | LLM zero macro | LLM zero micro | LLM few macro | LLM few micro |
|------|-----------:|-----------:|---------------:|---------------:|--------------:|--------------:|
| 0 | 0.401 | 0.767 | 0.138 | 0.154 | 0.175 | 0.169 |
| 1 | 0.466 | 0.768 | 0.140 | 0.155 | 0.178 | 0.173 |
| 2 | 0.443 | 0.757 | 0.142 | 0.157 | 0.175 | 0.171 |
| 3 | 0.416 | 0.766 | 0.142 | 0.157 | 0.174 | 0.170 |
| 4 | 0.395 | 0.755 | 0.141 | 0.156 | 0.175 | 0.170 |
| **Mean ± std** | **0.424 ± 0.030** | **0.763 ± 0.006** | **0.141 ± 0.001** | **0.156 ± 0.001** | **0.175 ± 0.002** | **0.171 ± 0.001** |

### Per-label mean F1 (5 folds)

| Label | BERT | LLM zero-shot | LLM few-shot |
|-------|-----:|--------------:|-------------:|
| toxic | 0.82 | 0.18 | 0.20 |
| obscene | 0.84 | 0.27 | **0.47** |
| insult | 0.74 | 0.26 | 0.21 |
| severe_toxic | 0.13 | 0.06 | 0.05 |
| threat | 0.00 | 0.02 | 0.03 |
| identity_hate | 0.02 | 0.06 | 0.11 |

Committed result files:

- `results/bert/` — per-fold metrics, `all_folds_metrics.csv`, `summary.json`
- `results/llm/zero_shot/` and `results/llm/few_shot/` — same layout
- `results/comparison/` — `comparison_summary.csv`, `per_fold_all_methods.csv`, `per_label_summary.csv`, PNG plots

---

## CLI reference

### `run_bert_cv.py`

```powershell
python scripts/run_bert_cv.py --folds all --skip-done
python scripts/run_bert_cv.py --folds 0,1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--folds` | `all` | Comma-separated fold indices or `all` |
| `--skip-done` | off | Skip folds with existing `fold_N_metrics.csv` |

Requires CUDA.

### `run_llm_cv.py`

```powershell
python scripts/run_llm_cv.py --folds all --modes zero_shot,few_shot --skip-done
python scripts/run_llm_cv.py --folds 0 --modes zero_shot --device cpu
```

| Flag | Default | Description |
|------|---------|-------------|
| `--folds` | `all` | Comma-separated fold indices or `all` |
| `--modes` | `zero_shot,few_shot` | Prompt mode(s) to run |
| `--skip-done` | off | Skip completed fold/mode pairs |
| `--device` | `cuda` | `cuda` or `cpu` |

---

## Git tracking

**Included:** source code, notebooks, `requirements.txt`, fold indices, experiment metrics (CSV/JSON), comparison outputs, assignment PDF.

**Excluded** (`.gitignore`): `.venv/`, `train.csv`, `models/`, BERT checkpoints, LLM cache, logs.

---

## References

1. Jigsaw / Google. *Toxic Comment Classification Challenge.* Kaggle, 2018.
2. Sanh, V. et al. *DistilBERT, a distilled version of BERT.* arXiv:1910.01108, 2019.
3. HuggingFace. *SmolLM2-1.7B-Instruct.* https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct
4. Sechidis, K. et al. *Stratification for multi-label data.* ECML PKDD, 2011.
