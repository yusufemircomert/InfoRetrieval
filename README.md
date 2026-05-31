# Toxic Comment Classification: Fine-Tuned BERT vs. Local LLM Prompting

**Course project — Fuzzy Logic**  
**Dataset:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)  
**Hardware:** NVIDIA GeForce RTX 5060 (CUDA 12.8)

---

## 1. Abstract

This project compares two approaches to **multi-label toxic comment classification** on the Jigsaw dataset:

1. **Supervised fine-tuning** of DistilBERT (`distilbert-base-uncased`)
2. **Prompt-based inference** with a local instruction-tuned LLM (`SmolLM2-1.7B-Instruct`), in zero-shot and few-shot settings

Both methods are evaluated with **5-fold cross-validation** on a stratified subsample of 25,000 comments, using identical train/validation/test splits. Fine-tuned DistilBERT substantially outperforms the local LLM on all metrics. The LLM exhibits high recall but very low precision, indicating a tendency to over-predict toxic labels.

---

## 2. Problem Statement

Online platforms need automated systems to detect harmful language. The Jigsaw dataset provides human-annotated comments with six **independent binary labels**:

| Label | Description |
|-------|-------------|
| `toxic` | Rude, disrespectful, or likely to drive users away |
| `severe_toxic` | Very hateful, aggressive, or extremely toxic |
| `obscene` | Obscenities or vulgar language |
| `threat` | Threat of violence |
| `insult` | Insulting or inflammatory language |
| `identity_hate` | Targets identity (race, religion, gender, etc.) |

This is a **multi-label** problem: a single comment may carry zero, one, or several labels simultaneously. Class imbalance is severe — rare labels such as `threat` and `identity_hate` appear in well under 1% of comments.

**Research question:** For a fixed compute budget on consumer GPU hardware, does fine-tuning a compact transformer outperform prompting a small local LLM without task-specific training?

---

## 3. Dataset and Preprocessing

### 3.1 Full dataset (EDA)

The raw training set (`train.csv`) contains approximately **159,571** comments. Label prevalence on the full corpus:

| Label | Count | Prevalence |
|-------|------:|-----------:|
| toxic | 15,294 | 9.58% |
| severe_toxic | 1,595 | 1.00% |
| obscene | 8,449 | 5.29% |
| threat | 478 | 0.30% |
| insult | 7,877 | 4.93% |
| identity_hate | 1,405 | 0.88% |

Roughly **16,225** comments (10.2%) carry at least one toxic label.

### 3.2 Stratified subsample

Training both BERT and the LLM on the full dataset is computationally expensive, especially for autoregressive LLM inference. We therefore draw a **25,000-row stratified subsample** using `MultilabelStratifiedShuffleSplit` (`iterative-stratification`), preserving the multi-label distribution. The subsample is saved to `data/train_subsampled.csv`.

### 3.3 Cross-validation splits

We use **5-fold multilabel stratified CV** (`MultilabelStratifiedKFold`, seed = 42). For each fold:

- **Train:** 16,000 comments (80% of subsample minus validation)
- **Validation:** 4,000 comments (20% of train portion, for early stopping)
- **Test:** 5,000 comments (held-out fold)

Test-set toxic rates are stable across folds (~10.0–10.3%). Fold indices are persisted under `data/fold_indices/` so BERT and LLM experiments use **identical splits**.

### 3.4 Text cleaning

Comments are lowercased implicitly by the tokenizer; whitespace is normalized (newlines collapsed, repeated spaces removed). No stemming or stop-word removal is applied.

---

## 4. Methods

### 4.1 Fine-tuned DistilBERT (supervised baseline)

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
| Precision | FP16 on GPU |

Each fold is trained independently; the best checkpoint (lowest validation loss) is selected for test evaluation.

### 4.2 Local LLM (zero-shot and few-shot)

| Setting | Value |
|---------|-------|
| Model | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| Temperature | 0.0 (greedy decoding) |
| Max new tokens | 128 |
| Output format | JSON with six 0/1 fields |

**Zero-shot prompt:** Label definitions + required JSON schema + comment text.

**Few-shot prompt:** Same structure plus three in-context examples (insult, benign, threat).

Responses are parsed as JSON with regex fallback. Predictions are **cached per fold and mode** under `results/llm/cache/` so runs can be stopped and resumed.

### 4.3 Evaluation metrics

Per-label **precision, recall, and F1** are computed with `sklearn`. Summary metrics:

- **Macro F1** — unweighted mean across labels (sensitive to rare classes)
- **Micro F1** — pooled over all label decisions (dominated by frequent labels)

Macro F1 is the primary metric for comparing methods on this imbalanced multi-label task.

---

## 5. Results

### 5.1 DistilBERT — 5-fold CV (complete)

| Fold | Macro F1 | Micro F1 |
|------|----------|----------|
| 0 | 0.401 | 0.767 |
| 1 | 0.466 | 0.768 |
| 2 | 0.443 | 0.757 |
| 3 | 0.416 | 0.766 |
| 4 | 0.395 | 0.755 |
| **Mean ± std** | **0.424 ± 0.031** | **0.763 ± 0.006** |

**Per-label mean F1 (across 5 folds):**

| Label | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| toxic | 0.83 | 0.81 | **0.82** |
| obscene | 0.86 | 0.82 | **0.84** |
| insult | 0.76 | 0.72 | **0.74** |
| severe_toxic | 0.27 | 0.09 | 0.14 |
| identity_hate | 0.20 | 0.01 | 0.02 |
| threat | 0.00 | 0.00 | **0.00** |

DistilBERT performs well on the three most frequent toxic categories (`toxic`, `obscene`, `insult`). It struggles severely on rare labels — especially `threat`, where no fold achieved non-zero F1. `identity_hate` and `severe_toxic` show similarly weak recall.

Micro F1 (~0.76) is much higher than macro F1 (~0.42) because the model correctly handles the abundant non-toxic majority and common labels, while rare labels drag down the macro average.

### 5.2 Local LLM — fold 0 only (partial)

Full 5-fold LLM evaluation is ongoing (inference is slow: on the order of hours per fold). Results below are from **fold 0** only.

| Method | Macro F1 | Micro F1 |
|--------|----------|----------|
| LLM zero-shot | 0.138 | 0.154 |
| LLM few-shot | 0.175 | 0.169 |
| BERT (fold 0) | 0.401 | 0.767 |

**Per-label F1 — fold 0 comparison:**

| Label | BERT | LLM zero-shot | LLM few-shot |
|-------|------|---------------|--------------|
| toxic | 0.83 | 0.18 | 0.19 |
| obscene | 0.84 | 0.27 | 0.46 |
| insult | 0.74 | 0.25 | 0.21 |
| severe_toxic | 0.00 | 0.06 | 0.05 |
| threat | 0.00 | 0.02 | 0.03 |
| identity_hate | 0.00 | 0.05 | 0.11 |

The LLM achieves **high recall** (often > 0.85) but **very low precision** (< 0.15) on most labels — it frequently assigns toxic labels to benign comments. Few-shot prompting improves obscene detection (F1 0.27 → 0.46) and identity_hate (0.05 → 0.11) but remains far below BERT.

### 5.3 Summary comparison

```
Method              Macro F1    Micro F1    Training required    Inference speed
─────────────────────────────────────────────────────────────────────────────────
DistilBERT (5-fold)   ~0.42       ~0.76     Yes (~30 min/fold)   Fast (batched)
LLM zero-shot (f0)    0.14        0.15      No                   Very slow
LLM few-shot (f0)     0.17        0.17      No                   Very slow
```

---

## 6. Discussion

### 6.1 Why BERT wins on this task

Fine-tuning adapts token representations directly to the Jigsaw label space. DistilBERT learns decision boundaries from thousands of labeled examples per fold, including subtle distinctions between insult and toxic language. The LLM, despite instruction tuning, was not trained on this specific taxonomy and must infer label boundaries from a short prompt.

### 6.2 LLM over-prediction

The local 1.7B model tends to mark comments as toxic across multiple labels simultaneously. This inflates recall but destroys precision. Possible causes:

- Small model capacity relative to the task complexity
- JSON-format pressure leading to default "1" assignments
- Lack of calibration — no probability scores, only hard 0/1 outputs
- Prompt length limits on long comments (truncation in generation context)

### 6.3 Rare-class failure mode

Both approaches fail on `threat`. BERT never predicts it; the LLM detects some threats but at unusable precision. This reflects extreme class imbalance (~0.3% prevalence) and the need for techniques such as class-weighted loss, focal loss, oversampling, or higher decision thresholds per label.

### 6.4 Practical recommendations

- **Production moderation:** Fine-tuned DistilBERT (or similar) is the better choice for accuracy and throughput.
- **Zero-shot exploration:** LLM prompting can prototype new label definitions without retraining, but requires larger models or API access for competitive quality.
- **Fair comparison:** Shared CV folds and cached LLM outputs ensure reproducible, methodologically sound evaluation.

---

## 7. Project Structure

```
final/
├── train.csv                    # Full Jigsaw training data
├── data/
│   ├── train_subsampled.csv     # 25k stratified subsample
│   └── fold_indices/            # Shared 5-fold splits
├── src/
│   ├── config.py                # Hyperparameters and paths
│   ├── data_utils.py            # Loading, subsampling, CV splits
│   ├── bert_model.py            # DistilBERT training and inference
│   ├── llm_inference.py         # Local LLM classifier with caching
│   ├── llm_prompts.py           # Zero/few-shot templates and parsing
│   ├── metrics.py               # Precision, recall, F1
│   └── results_io.py            # CSV/JSON export
├── notebooks/
│   ├── 01_eda_and_splits.ipynb
│   ├── 02_bert_cv.ipynb
│   └── 03_llm_cv.ipynb
├── scripts/
│   ├── run_bert_cv.py
│   ├── run_llm_cv.py
│   └── run_llm_overnight.ps1    # Long-running LLM jobs
└── results/
    ├── bert/                    # Checkpoints and metrics (5 folds complete)
    └── llm/                     # zero_shot/, few_shot/, cache/
```

---

## 8. Reproducibility

### Environment

```powershell
# From project root (final/)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# PyTorch with CUDA 12.8 (RTX 5060)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Running experiments

```powershell
# EDA and fold creation
jupyter notebook notebooks/01_eda_and_splits.ipynb

# BERT — all 5 folds
python scripts/run_bert_cv.py

# LLM — resume-safe, one fold at a time
python scripts/run_llm_cv.py --folds 0 --modes zero_shot few_shot
```

Random seed **42** is fixed for subsampling and fold creation. LLM caches prevent redundant inference when restarting.

---

## 9. Conclusion

We built a reproducible pipeline to compare supervised transformer fine-tuning against local LLM prompting for multi-label toxic comment classification. On identical 5-fold splits:

- **DistilBERT** achieves macro F1 ≈ **0.42** and strong performance on common labels.
- **SmolLM2-1.7B** (fold 0) reaches macro F1 ≈ **0.14–0.17**, with few-shot prompting offering modest gains over zero-shot.
- Both methods fail on the rarest labels, highlighting class imbalance as the main remaining challenge.

Future work: complete the remaining LLM folds, experiment with class-weighted training, per-label thresholds, and larger LLMs or chain-of-thought prompting for rare categories.

---

## 10. References

1. Jigsaw / Google. *Toxic Comment Classification Challenge.* Kaggle, 2018.
2. Sanh, V. et al. *DistilBERT, a distilled version of BERT.* arXiv:1910.01108, 2019.
3. HuggingFace. *SmolLM2.* https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct
4. Sechidis, K. et al. *Stratification for multi-label data.* ECML PKDD, 2011.

---

