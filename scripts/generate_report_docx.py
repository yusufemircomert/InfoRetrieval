"""Generate REPORT.docx from project report content."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt
from docx.oxml.ns import qn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "REPORT.docx"


def set_default_font(doc: Document, name: str = "Calibri", size: Pt = Pt(11)) -> None:
    style = doc.styles["Normal"]
    style.font.name = name
    style.font.size = size
    style._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def add_title_page(doc: Document) -> None:
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(
        "Toxic Comment Classification:\nFine-Tuned BERT vs. Local LLM Prompting"
    )
    run.bold = True
    run.font.size = Pt(20)

    doc.add_paragraph()
    for line in (
        "Course project — Fuzzy Logic",
        "Dataset: Jigsaw Toxic Comment Classification Challenge",
        "Hardware: NVIDIA GeForce RTX 5060 (CUDA 12.8)",
    ):
        p = doc.add_paragraph(line)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_page_break()


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def add_paragraph(doc: Document, text: str, bold: bool = False) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def add_numbered(doc: Document, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Number")


def add_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    hdr_cells = table.rows[0].cells
    for i, header in enumerate(headers):
        hdr_cells[i].text = header
        for paragraph in hdr_cells[i].paragraphs:
            for run in paragraph.runs:
                run.bold = True
    for row_idx, row in enumerate(rows, start=1):
        for col_idx, value in enumerate(row):
            table.rows[row_idx].cells[col_idx].text = value
    doc.add_paragraph()


def add_code_block(doc: Document, text: str) -> None:
    for line in text.strip().splitlines():
        p = doc.add_paragraph()
        run = p.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(9)


def build_report() -> Document:
    doc = Document()
    set_default_font(doc)
    add_title_page(doc)

    add_heading(doc, "1. Abstract")
    add_paragraph(
        doc,
        "This project compares two approaches to multi-label toxic comment classification "
        "on the Jigsaw dataset:",
    )
    add_numbered(
        doc,
        [
            "Supervised fine-tuning of DistilBERT (distilbert-base-uncased)",
            "Prompt-based inference with a local instruction-tuned LLM "
            "(SmolLM2-1.7B-Instruct), in zero-shot and few-shot settings",
        ],
    )
    add_paragraph(
        doc,
        "Both methods are evaluated with 5-fold cross-validation on a stratified subsample "
        "of 25,000 comments, using identical train/validation/test splits. Fine-tuned DistilBERT "
        "substantially outperforms the local LLM on all metrics. The LLM exhibits high recall "
        "but very low precision, indicating a tendency to over-predict toxic labels.",
    )

    add_heading(doc, "2. Problem Statement")
    add_paragraph(
        doc,
        "Online platforms need automated systems to detect harmful language. The Jigsaw dataset "
        "provides human-annotated comments with six independent binary labels:",
    )
    add_table(
        doc,
        ["Label", "Description"],
        [
            ["toxic", "Rude, disrespectful, or likely to drive users away"],
            ["severe_toxic", "Very hateful, aggressive, or extremely toxic"],
            ["obscene", "Obscenities or vulgar language"],
            ["threat", "Threat of violence"],
            ["insult", "Insulting or inflammatory language"],
            ["identity_hate", "Targets identity (race, religion, gender, etc.)"],
        ],
    )
    add_paragraph(
        doc,
        "This is a multi-label problem: a single comment may carry zero, one, or several labels "
        "simultaneously. Class imbalance is severe — rare labels such as threat and identity_hate "
        "appear in well under 1% of comments.",
    )
    add_paragraph(
        doc,
        "Research question: For a fixed compute budget on consumer GPU hardware, does fine-tuning "
        "a compact transformer outperform prompting a small local LLM without task-specific training?",
        bold=True,
    )

    add_heading(doc, "3. Dataset and Preprocessing")
    add_heading(doc, "3.1 Full dataset (EDA)", level=2)
    add_paragraph(
        doc,
        "The raw training set (train.csv) contains approximately 159,571 comments. "
        "Label prevalence on the full corpus:",
    )
    add_table(
        doc,
        ["Label", "Count", "Prevalence"],
        [
            ["toxic", "15,294", "9.58%"],
            ["severe_toxic", "1,595", "1.00%"],
            ["obscene", "8,449", "5.29%"],
            ["threat", "478", "0.30%"],
            ["insult", "7,877", "4.93%"],
            ["identity_hate", "1,405", "0.88%"],
        ],
    )
    add_paragraph(doc, "Roughly 16,225 comments (10.2%) carry at least one toxic label.")

    add_heading(doc, "3.2 Stratified subsample", level=2)
    add_paragraph(
        doc,
        "Training both BERT and the LLM on the full dataset is computationally expensive, "
        "especially for autoregressive LLM inference. We therefore draw a 25,000-row stratified "
        "subsample using MultilabelStratifiedShuffleSplit (iterative-stratification), preserving "
        "the multi-label distribution. The subsample is saved to data/train_subsampled.csv.",
    )

    add_heading(doc, "3.3 Cross-validation splits", level=2)
    add_paragraph(
        doc,
        "We use 5-fold multilabel stratified CV (MultilabelStratifiedKFold, seed = 42). "
        "For each fold:",
    )
    add_bullets(
        doc,
        [
            "Train: 16,000 comments (80% of subsample minus validation)",
            "Validation: 4,000 comments (20% of train portion, for early stopping)",
            "Test: 5,000 comments (held-out fold)",
        ],
    )
    add_paragraph(
        doc,
        "Test-set toxic rates are stable across folds (~10.0–10.3%). Fold indices are persisted "
        "under data/fold_indices/ so BERT and LLM experiments use identical splits.",
    )

    add_heading(doc, "3.4 Text cleaning", level=2)
    add_paragraph(
        doc,
        "Comments are lowercased implicitly by the tokenizer; whitespace is normalized "
        "(newlines collapsed, repeated spaces removed). No stemming or stop-word removal is applied.",
    )

    add_heading(doc, "4. Methods")
    add_heading(doc, "4.1 Fine-tuned DistilBERT (supervised baseline)", level=2)
    add_table(
        doc,
        ["Hyperparameter", "Value"],
        [
            ["Model", "distilbert-base-uncased (local copy in models/)"],
            ["Task head", "Multi-label classification (sigmoid + BCE)"],
            ["Max sequence length", "256 tokens"],
            ["Epochs", "3"],
            ["Batch size", "16"],
            ["Learning rate", "2e-5"],
            ["Early stopping", "Patience = 1 epoch on validation loss"],
            ["Threshold", "0.5 on sigmoid outputs"],
            ["Precision", "FP16 on GPU"],
        ],
    )
    add_paragraph(
        doc,
        "Each fold is trained independently; the best checkpoint (lowest validation loss) "
        "is selected for test evaluation.",
    )

    add_heading(doc, "4.2 Local LLM (zero-shot and few-shot)", level=2)
    add_table(
        doc,
        ["Setting", "Value"],
        [
            ["Model", "HuggingFaceTB/SmolLM2-1.7B-Instruct"],
            ["Temperature", "0.0 (greedy decoding)"],
            ["Max new tokens", "128"],
            ["Output format", "JSON with six 0/1 fields"],
        ],
    )
    add_paragraph(doc, "Zero-shot prompt: Label definitions + required JSON schema + comment text.")
    add_paragraph(
        doc,
        "Few-shot prompt: Same structure plus three in-context examples (insult, benign, threat).",
    )
    add_paragraph(
        doc,
        "Responses are parsed as JSON with regex fallback. Predictions are cached per fold and mode "
        "under results/llm/cache/ so runs can be stopped and resumed.",
    )

    add_heading(doc, "4.3 Evaluation metrics", level=2)
    add_paragraph(doc, "Per-label precision, recall, and F1 are computed with scikit-learn.")
    add_bullets(
        doc,
        [
            "Macro F1 — unweighted mean across labels (sensitive to rare classes)",
            "Micro F1 — pooled over all label decisions (dominated by frequent labels)",
        ],
    )
    add_paragraph(
        doc,
        "Macro F1 is the primary metric for comparing methods on this imbalanced multi-label task.",
    )

    add_heading(doc, "5. Results")
    add_heading(doc, "5.1 DistilBERT — 5-fold CV (complete)", level=2)
    add_table(
        doc,
        ["Fold", "Macro F1", "Micro F1"],
        [
            ["0", "0.401", "0.767"],
            ["1", "0.466", "0.768"],
            ["2", "0.443", "0.757"],
            ["3", "0.416", "0.766"],
            ["4", "0.395", "0.755"],
            ["Mean ± std", "0.424 ± 0.031", "0.763 ± 0.006"],
        ],
    )
    add_paragraph(doc, "Per-label mean F1 (across 5 folds):", bold=True)
    add_table(
        doc,
        ["Label", "Precision", "Recall", "F1"],
        [
            ["toxic", "0.83", "0.81", "0.82"],
            ["obscene", "0.86", "0.82", "0.84"],
            ["insult", "0.76", "0.72", "0.74"],
            ["severe_toxic", "0.27", "0.09", "0.14"],
            ["identity_hate", "0.20", "0.01", "0.02"],
            ["threat", "0.00", "0.00", "0.00"],
        ],
    )
    add_paragraph(
        doc,
        "DistilBERT performs well on the three most frequent toxic categories (toxic, obscene, insult). "
        "It struggles severely on rare labels — especially threat, where no fold achieved non-zero F1.",
    )

    add_heading(doc, "5.2 Local LLM — fold 0 only (partial)", level=2)
    add_paragraph(
        doc,
        "Full 5-fold LLM evaluation is ongoing (inference is slow: on the order of hours per fold). "
        "Results below are from fold 0 only.",
    )
    add_table(
        doc,
        ["Method", "Macro F1", "Micro F1"],
        [
            ["LLM zero-shot", "0.138", "0.154"],
            ["LLM few-shot", "0.175", "0.169"],
            ["BERT (fold 0)", "0.401", "0.767"],
        ],
    )
    add_paragraph(doc, "Per-label F1 — fold 0 comparison:", bold=True)
    add_table(
        doc,
        ["Label", "BERT", "LLM zero-shot", "LLM few-shot"],
        [
            ["toxic", "0.83", "0.18", "0.19"],
            ["obscene", "0.84", "0.27", "0.46"],
            ["insult", "0.74", "0.25", "0.21"],
            ["severe_toxic", "0.00", "0.06", "0.05"],
            ["threat", "0.00", "0.02", "0.03"],
            ["identity_hate", "0.00", "0.05", "0.11"],
        ],
    )
    add_paragraph(
        doc,
        "The LLM achieves high recall (often > 0.85) but very low precision (< 0.15) on most labels. "
        "Few-shot prompting improves obscene detection (F1 0.27 → 0.46) and identity_hate "
        "(0.05 → 0.11) but remains far below BERT.",
    )

    add_heading(doc, "5.3 Summary comparison", level=2)
    add_table(
        doc,
        ["Method", "Macro F1", "Micro F1", "Training required", "Inference speed"],
        [
            ["DistilBERT (5-fold)", "~0.42", "~0.76", "Yes (~30 min/fold)", "Fast (batched)"],
            ["LLM zero-shot (f0)", "0.14", "0.15", "No", "Very slow"],
            ["LLM few-shot (f0)", "0.17", "0.17", "No", "Very slow"],
        ],
    )

    add_heading(doc, "6. Discussion")
    add_heading(doc, "6.1 Why BERT wins on this task", level=2)
    add_paragraph(
        doc,
        "Fine-tuning adapts token representations directly to the Jigsaw label space. DistilBERT "
        "learns decision boundaries from thousands of labeled examples per fold. The LLM was not "
        "trained on this specific taxonomy and must infer label boundaries from a short prompt.",
    )
    add_heading(doc, "6.2 LLM over-prediction", level=2)
    add_bullets(
        doc,
        [
            "Small model capacity relative to the task complexity",
            "JSON-format pressure leading to default \"1\" assignments",
            "Lack of calibration — no probability scores, only hard 0/1 outputs",
            "Prompt length limits on long comments",
        ],
    )
    add_heading(doc, "6.3 Rare-class failure mode", level=2)
    add_paragraph(
        doc,
        "Both approaches fail on threat. This reflects extreme class imbalance (~0.3% prevalence) "
        "and the need for class-weighted loss, focal loss, oversampling, or per-label thresholds.",
    )
    add_heading(doc, "6.4 Practical recommendations", level=2)
    add_bullets(
        doc,
        [
            "Production moderation: Fine-tuned DistilBERT is the better choice for accuracy and throughput.",
            "Zero-shot exploration: LLM prompting can prototype new labels without retraining.",
            "Fair comparison: Shared CV folds and cached LLM outputs ensure reproducible evaluation.",
        ],
    )

    add_heading(doc, "7. Project Structure")
    add_code_block(
        doc,
        """final/
├── train.csv
├── data/ (subsample + fold indices)
├── src/ (config, data_utils, bert_model, llm_inference, metrics)
├── notebooks/ (EDA, BERT CV, LLM CV)
├── scripts/ (run_bert_cv.py, run_llm_cv.py)
└── results/ (bert/, llm/)""",
    )

    add_heading(doc, "8. Reproducibility")
    add_heading(doc, "Environment", level=2)
    add_code_block(
        doc,
        """python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt""",
    )
    add_heading(doc, "Running experiments", level=2)
    add_code_block(
        doc,
        """jupyter notebook notebooks/01_eda_and_splits.ipynb
python scripts/run_bert_cv.py
python scripts/run_llm_cv.py --folds 0 --modes zero_shot few_shot""",
    )
    add_paragraph(doc, "Random seed 42 is fixed for subsampling and fold creation.")

    add_heading(doc, "9. Conclusion")
    add_bullets(
        doc,
        [
            "DistilBERT achieves macro F1 ≈ 0.42 and strong performance on common labels.",
            "SmolLM2-1.7B (fold 0) reaches macro F1 ≈ 0.14–0.17; few-shot helps modestly.",
            "Both methods fail on the rarest labels, highlighting class imbalance as the main challenge.",
        ],
    )
    add_paragraph(
        doc,
        "Future work: complete remaining LLM folds, class-weighted training, per-label thresholds, "
        "and larger LLMs for rare categories.",
    )

    add_heading(doc, "10. References")
    add_numbered(
        doc,
        [
            "Jigsaw / Google. Toxic Comment Classification Challenge. Kaggle, 2018.",
            "Sanh, V. et al. DistilBERT, a distilled version of BERT. arXiv:1910.01108, 2019.",
            "HuggingFace. SmolLM2. https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "Sechidis, K. et al. Stratification for multi-label data. ECML PKDD, 2011.",
        ],
    )

    return doc


def main() -> None:
    doc = build_report()
    doc.save(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
