"""Local LLM inference using Hugging Face transformers (no Ollama required)."""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    LABELS,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    RESULTS_DIR,
)
from src.llm_prompts import (
    build_few_shot_prompt,
    build_zero_shot_prompt,
    parse_with_regex_fallback,
    predictions_dict_to_array,
)
from src.metrics import evaluate_predictions


PromptMode = Literal["zero_shot", "few_shot"]


@dataclass
class LLMFoldResult:
    fold_idx: int
    mode: PromptMode
    metrics: dict
    y_true: np.ndarray
    y_pred: np.ndarray


def _comment_cache_key(comment: str, mode: PromptMode) -> str:
    payload = f"{mode}\n{comment}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_cache(cache_path: Path) -> dict[str, dict[str, int]]:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache_path: Path, cache: dict[str, dict[str, int]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def default_cache_path(fold_idx: int, mode: PromptMode) -> Path:
    return RESULTS_DIR / "llm" / "cache" / f"fold_{fold_idx}_{mode}.json"


def clear_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class LocalLLMClassifier:
    def __init__(
        self,
        model_name: str = LLM_MODEL_NAME,
        device: str | None = None,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if device == "cpu" or device == -1:
            self.use_cuda = False
            self.device = torch.device("cpu")
        elif device is None:
            self.use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if self.use_cuda else "cpu")
        else:
            self.use_cuda = True
            self.device = torch.device("cuda")

        dtype = torch.float16 if self.use_cuda else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def _build_prompt(self, comment: str, mode: PromptMode) -> str:
        if mode == "zero_shot":
            return build_zero_shot_prompt(comment)
        return build_few_shot_prompt(comment)

    def classify_comment(self, comment: str, mode: PromptMode = "zero_shot") -> dict[str, int]:
        prompt = self._build_prompt(comment, mode)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        generated = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return parse_with_regex_fallback(generated)

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        mode: PromptMode = "zero_shot",
        cache_path: Path | None = None,
        max_samples: int | None = None,
        save_every: int = 10,
        clear_cache_every: int = 50,
    ) -> np.ndarray:
        texts = df["comment_text"].tolist()
        if max_samples is not None:
            texts = texts[:max_samples]

        cache: dict[str, dict[str, int]] = {}
        if cache_path is not None:
            cache = _load_cache(cache_path)

        predictions: list[list[int]] = []
        pending_saves = 0
        new_inferences = 0

        for idx, comment in enumerate(tqdm(texts, desc=f"LLM {mode}")):
            key = _comment_cache_key(comment, mode)
            if key in cache:
                parsed = cache[key]
            else:
                parsed = self.classify_comment(comment, mode=mode)
                cache[key] = parsed
                pending_saves += 1
                new_inferences += 1

            predictions.append(predictions_dict_to_array(parsed))

            if cache_path is not None and pending_saves >= save_every:
                _save_cache(cache_path, cache)
                pending_saves = 0

            if self.use_cuda and new_inferences > 0 and new_inferences % clear_cache_every == 0:
                clear_gpu_memory()

        if cache_path is not None:
            _save_cache(cache_path, cache)

        if self.use_cuda:
            clear_gpu_memory()

        return np.asarray(predictions, dtype=int)

    def unload(self) -> None:
        del self.model
        del self.tokenizer
        clear_gpu_memory()


def evaluate_llm_fold(
    classifier: LocalLLMClassifier,
    fold_idx: int,
    test_df: pd.DataFrame,
    mode: PromptMode,
    cache_path: Path | None = None,
    max_samples: int | None = None,
) -> LLMFoldResult:
    subset = test_df
    if max_samples is not None:
        subset = test_df.head(max_samples)

    resolved_cache = cache_path or default_cache_path(fold_idx, mode)
    y_true = subset[LABELS].to_numpy()
    y_pred = classifier.classify_dataframe(
        subset,
        mode=mode,
        cache_path=resolved_cache,
    )

    metrics = evaluate_predictions(y_true, y_pred)
    return LLMFoldResult(
        fold_idx=fold_idx,
        mode=mode,
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
    )
