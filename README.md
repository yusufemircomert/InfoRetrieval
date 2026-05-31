# Toxic Comment Classification: Fine-Tuned BERT vs. Local LLM Prompting

**Course:** AIN 428 — Information Retrieval (final project)  
**Dataset:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)  
**Hardware tested:** NVIDIA GeForce RTX 5060 GPU (8 GB VRAM, CUDA 12.8)

Multi-label toxic comment classification comparing **supervised DistilBERT fine-tuning** against **local LLM prompting** (zero-shot and few-shot), evaluated with **5-fold cross-validation** on identical stratified splits.

---

## Results at a glance

| Method | Macro F1 (mean ± std) | Micro F1 (mean ± std) |
|--------|----------------------:|----------------------:|
| **DistilBERT** (5-fold CV) | **0.424 ± 0.030** | **0.763 ± 0.006** |
| LLM zero-shot (5-fold CV) | 0.141 ± 0.001 | 0.156 ± 0.001 |
| LLM few-shot (5-fold CV) | 0.175 ± 0.002 | 0.171 ± 0.001 |

Fine-tuned DistilBERT clearly outperforms the local 1.7B LLM. The LLM achieves high recall but very low precision — it over-predicts toxic labels. Both approaches struggle on rare classes (`threat`, `identity_hate`).

Detailed tables, per-label breakdowns, and plots: run `notebooks/04_comparison.ipynb` or see `results/comparison/`.

---

## Quick start

### 1. Clone and create environment

```powershell
cd path\to\InfoRetrieval   # or your clone of this repo

python -m venv .venv
.\.venv\Scripts\Activate.ps1

# PyTorch with CUDA 12.8 (required for RTX 50-series / sm_120)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Optional: Word report export
pip install python-docx
```

Register the Jupyter kernel (optional):

```powershell
python -m ipykernel install --user --name fuzzy-final --display-name "Python 3.12 (fuzzy-final)"
```

### 2. Download data and models (not in Git)

These large files are excluded by `.gitignore` and must be obtained locally:

| Item | Location | How to get it |
|------|----------|---------------|
| Jigsaw training set | `train.csv` | Download from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) |
| DistilBERT weights | `models/distilbert-base-uncased/` | `huggingface-cli download distilbert-base-uncased --local-dir models/distilbert-base-uncased` |
| Subsample & folds | `data/train_subsampled.csv`, `data/fold_indices/` | Run `notebooks/01_eda_and_splits.ipynb` (fold indices are committed; subsample is regenerated) |

The LLM (`HuggingFaceTB/SmolLM2-1.7B-Instruct`) is downloaded automatically on first inference.

### 3. Run experiments

```powershell
# Sanity check: subsample, folds, prompt parsing
python scripts/smoke_test.py

# BERT — all 5 folds (GPU, ~30 min/fold)
python scripts/run_bert_cv.py --folds all --skip-done

# LLM — zero-shot + few-shot (GPU, several hours total; resume-safe)
python scripts/run_llm_cv.py --folds all --modes zero_shot,few_shot --skip-done

# Comparison tables and plots
jupyter notebook notebooks/04_comparison.ipynb

```

---

## Problem statement

Online platforms need automated moderation. The Jigsaw dataset provides six **independent binary labels** per comment:

| Label | Description |
|-------|-------------|
| `toxic` | Rude, disrespectful, or likely to drive users away |
| `severe_toxic` | Very hateful, aggressive, or extremely toxic |
| `obscene` | Obscenities or vulgar language |
| `threat` | Threat of violence |
| `insult` | Insulting or inflammatory language |
| `identity_hate` | Targets identity (race, religion, gender, etc.) |

This is **multi-label** classification with severe class imbalance (e.g. `threat` ≈ 0.3% of comments).

**Research question:** On consumer GPU hardware, does fine-tuning a compact transformer outperform prompting a small local LLM without task-specific training?

---

## Dataset and preprocessing

### Full dataset (EDA)

`train.csv` contains ~**159,571** comments. Label prevalence on the full corpus:

| Label | Count | Prevalence |
|-------|------:|-----------:|
| toxic | 15,294 | 9.58% |
| severe_toxic | 1,595 | 1.00% |
| obscene | 8,449 | 5.29% |
| threat | 478 | 0.30% |
| insult | 7,877 | 4.93% |
| identity_hate | 1,405 | 0.88% |

~**16,225** comments (10.2%) carry at least one toxic label. See `notebooks/01_eda_and_splits.ipynb`.

### Stratified subsample

A **25,000-row stratified subsample** is drawn with `MultilabelStratifiedShuffleSplit` (`iterative-stratification`, seed = 42), preserving multi-label distribution. Saved to `data/train_subsampled.csv`.

### Cross-validation splits

**5-fold multilabel stratified CV** (`MultilabelStratifiedKFold`, seed = 42). Per fold:

- **Train:** 16,000 comments
- **Validation:** 4,000 comments (early stopping)
- **Test:** 5,000 comments (held-out fold)

Fold indices live in `data/fold_indices/` (`.npy` arrays + `metadata.json`). BERT and LLM experiments use **identical splits**.

---

## Methods

### Fine-tuned DistilBERT

| Hyperparameter | Value |
|----------------|-------|
| Model | `distilbert-base-uncased` (local copy in `models/`) |
| Task head | Multi-label classification (sigmoid + BCE) |
| Max sequence length | 256 tokens |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Early stopping | Patience = 1 epoch on validation loss |
| Threshold | 0.5 on sigmoid outputs |
| Mixed precision | FP16 on GPU |

Each fold is trained independently; the best checkpoint (lowest validation loss) is evaluated on the test fold. Checkpoints are saved under `results/bert/checkpoints/` (gitignored).

### Local LLM (zero-shot and few-shot)

| Setting | Value |
|---------|-------|
| Model | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| Temperature | 0.0 (greedy decoding) |
| Max new tokens | 64 |
| Output format | JSON with six 0/1 fields |

- **Zero-shot:** Label definitions + JSON schema + comment text.
- **Few-shot:** Same structure plus three in-context examples (insult, benign, threat).

Responses are parsed as JSON with regex fallback. Raw predictions are **cached** per fold and mode under `results/llm/cache/` so runs can be stopped and resumed. The classifier loads and unloads the model per fold to reduce VRAM pressure.

### Evaluation metrics

Per-label **precision, recall, and F1** via `sklearn`. Summary metrics:

- **Macro F1** — unweighted mean across labels (primary metric for imbalanced multi-label tasks)
- **Micro F1** — pooled over all label decisions

---

## Results (complete — all 5 folds)

### DistilBERT

| Fold | Macro F1 | Micro F1 |
|------|----------|----------|
| 0 | 0.401 | 0.767 |
| 1 | 0.466 | 0.768 |
| 2 | 0.443 | 0.757 |
| 3 | 0.416 | 0.766 |
| 4 | 0.395 | 0.755 |
| **Mean ± std** | **0.424 ± 0.030** | **0.763 ± 0.006** |

**Per-label mean F1 (5 folds):**

| Label | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| toxic | 0.83 | 0.81 | **0.82** |
| obscene | 0.86 | 0.82 | **0.84** |
| insult | 0.76 | 0.72 | **0.74** |
| severe_toxic | 0.27 | 0.09 | 0.13 |
| identity_hate | 0.20 | 0.01 | 0.02 |
| threat | 0.00 | 0.00 | **0.00** |

DistilBERT performs well on frequent labels but fails on `threat` (zero F1 in every fold) and barely detects `identity_hate`.

### LLM zero-shot (5 folds)

| Fold | Macro F1 | Micro F1 |
|------|----------|----------|
| 0 | 0.138 | 0.154 |
| 1 | 0.140 | 0.155 |
| 2 | 0.142 | 0.157 |
| 3 | 0.142 | 0.157 |
| 4 | 0.141 | 0.156 |
| **Mean ± std** | **0.141 ± 0.001** | **0.156 ± 0.001** |

### LLM few-shot (5 folds)

| Fold | Macro F1 | Micro F1 |
|------|----------|----------|
| 0 | 0.175 | 0.169 |
| 1 | 0.178 | 0.173 |
| 2 | 0.175 | 0.171 |
| 3 | 0.174 | 0.170 |
| 4 | 0.175 | 0.170 |
| **Mean ± std** | **0.175 ± 0.002** | **0.171 ± 0.001** |

Few-shot prompting improves macro F1 by ~0.035 over zero-shot but remains far below BERT.

### Per-label F1 — mean across 5 folds

| Label | BERT | LLM zero-shot | LLM few-shot |
|-------|-----:|--------------:|-------------:|
| toxic | 0.82 | 0.18 | 0.20 |
| obscene | 0.84 | 0.27 | **0.47** |
| insult | 0.74 | 0.26 | 0.21 |
| severe_toxic | 0.13 | 0.06 | 0.05 |
| threat | 0.00 | 0.02 | 0.03 |
| identity_hate | 0.02 | 0.06 | 0.11 |

The LLM shows **recall > 0.80** on most labels but **precision < 0.15** (except few-shot `obscene` at ~0.34). Few-shot helps most on `obscene` and `identity_hate`.

### Method comparison

```
Method                 Macro F1    Micro F1    Training     Inference
────────────────────────────────────────────────────────────────────────
DistilBERT (5-fold)      0.424       0.763       Yes          Fast (batched)
LLM zero-shot (5-fold)   0.141       0.156       No           Very slow
LLM few-shot (5-fold)    0.175       0.171       No           Very slow
```

Exported artifacts: `results/comparison/comparison_summary.csv`, `per_fold_all_methods.csv`, `per_label_summary.csv`, and PNG plots from notebook 04.

---

## Discussion

**Why BERT wins:** Fine-tuning adapts representations directly to the Jigsaw label space from thousands of labeled examples per fold. The 1.7B LLM must infer boundaries from a short prompt alone.

**LLM over-prediction:** The model frequently assigns multiple toxic labels to benign comments, inflating recall and destroying precision. Likely factors include small model capacity, JSON-format pressure, and lack of calibrated probabilities.

**Rare classes:** Both methods fail on `threat`. BERT never predicts it; the LLM detects some instances at unusable precision. Addressing this would require class-weighted loss, oversampling, or per-label thresholds.

**Reproducibility:** Shared CV folds, fixed seed (42), and LLM response caching ensure fair comparison and resumable runs.

---

## Project structure

```
.
├── train.csv                         # Kaggle dataset (not in Git — download locally)
├── requirements.txt
├── .gitignore
├── .gitattributes
├── data/
│   ├── train_subsampled.csv          # Regenerated from train.csv (not in Git)
│   └── fold_indices/                 # Shared 5-fold splits (.npy + metadata.json)
├── models/
│   └── distilbert-base-uncased/      # Local BERT weights (not in Git — ~1.5 GB)
├── src/
│   ├── config.py                     # Paths, hyperparameters, model names
│   ├── data_utils.py                 # Loading, subsampling, CV splits
│   ├── bert_model.py                 # DistilBERT training and evaluation
│   ├── llm_inference.py              # Local LLM classifier with caching
│   ├── llm_prompts.py                # Zero/few-shot templates and JSON parsing
│   ├── metrics.py                    # Precision, recall, F1 aggregation
│   └── results_io.py                 # CSV/JSON export helpers
├── notebooks/
│   ├── 01_eda_and_splits.ipynb       # EDA, subsample, fold creation
│   ├── 02_bert_cv.ipynb              # Interactive BERT CV (optional)
│   ├── 03_llm_cv.ipynb               # Interactive LLM CV (fold 0 + caching demo)
│   └── 04_comparison.ipynb           # Tables, plots, export to results/comparison/
├── scripts/
│   ├── run_bert_cv.py                # CLI: BERT 5-fold CV
│   ├── run_llm_cv.py                 # CLI: LLM zero/few-shot CV
│   ├── smoke_test.py                 # Quick pipeline sanity check
│   ├── run_overnight.ps1             # Wrapper for long BERT runs
│   ├── run_llm_overnight.ps1         # Wrapper for long LLM runs
│   └── run_llm_overnight.bat         # Batch wrapper for LLM runs
└── results/
    ├── bert/                         # Metrics CSVs + summary.json (checkpoints gitignored)
    ├── llm/
    │   ├── zero_shot/                # Per-fold and consolidated metrics
    │   ├── few_shot/
    │   └── cache/                    # Raw LLM responses (gitignored, regeneratable)
    └── comparison/                   # Cross-method summary CSVs and plots
```

---

## CLI reference

### `run_bert_cv.py`

```powershell
python scripts/run_bert_cv.py --folds all --skip-done
python scripts/run_bert_cv.py --folds 0,1          # specific folds only
```

| Flag | Description |
|------|-------------|
| `--folds` | Comma-separated fold indices or `all` (default: `all`) |
| `--skip-done` | Skip folds that already have `results/bert/fold_N_metrics.csv` |

Requires CUDA; fails fast if GPU is unavailable.

### `run_llm_cv.py`

```powershell
python scripts/run_llm_cv.py --folds all --modes zero_shot,few_shot --skip-done
python scripts/run_llm_cv.py --folds 0 --modes zero_shot --device cpu   # if GPU busy
```

| Flag | Description |
|------|-------------|
| `--folds` | Comma-separated indices or `all` (default: `1,2,3,4`) |
| `--modes` | `zero_shot`, `few_shot`, or both (default: both) |
| `--skip-done` | Skip fold/mode pairs with existing metrics CSV |
| `--device` | `cuda` (default) or `cpu` |

---

## What is tracked in Git

**Included:** source code, notebooks, `requirements.txt`, fold indices, metric CSVs/JSON, comparison outputs.

**Excluded** (see `.gitignore`): `.venv/`, `train.csv`, `models/`, BERT checkpoints, LLM cache, logs, generated `*.docx`.

---

## Conclusion

On identical 5-fold splits of a 25,000-comment stratified subsample:

- **DistilBERT** reaches macro F1 **0.424** with strong performance on `toxic`, `obscene`, and `insult`.
- **SmolLM2-1.7B** reaches macro F1 **0.141** (zero-shot) and **0.175** (few-shot), with high recall but poor precision.
- **Rare labels** remain the main challenge for both approaches.

For production moderation on limited hardware, fine-tuned DistilBERT is the better choice. LLM prompting may still be useful for rapid prototyping of new label definitions without retraining.

---

## References

1. Jigsaw / Google. *Toxic Comment Classification Challenge.* Kaggle, 2018.
2. Sanh, V. et al. *DistilBERT, a distilled version of BERT.* arXiv:1910.01108, 2019.
3. HuggingFace. *SmolLM2-1.7B-Instruct.* https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct
4. Sechidis, K. et al. *Stratification for multi-label data.* ECML PKDD, 2011.
