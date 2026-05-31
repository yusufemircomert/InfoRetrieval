"""Prompt templates and output parsing for local LLM evaluation."""

from __future__ import annotations

import json
import re
from typing import Any

from src.config import LABELS

LABEL_DESCRIPTIONS = {
    "toxic": "rude, disrespectful, or likely to make someone leave a discussion",
    "severe_toxic": "very hateful, aggressive, or extremely toxic",
    "obscene": "contains obscenities or vulgar language",
    "threat": "contains a threat of violence",
    "insult": "contains insulting or inflammatory language",
    "identity_hate": "targets someone because of identity (race, religion, gender, etc.)",
}


def _label_block() -> str:
    lines = [f'- "{label}": {LABEL_DESCRIPTIONS[label]}' for label in LABELS]
    return "\n".join(lines)


def _json_template() -> str:
    template = {label: 0 for label in LABELS}
    return json.dumps(template)


ZERO_SHOT_PROMPT = """You are a toxicity classifier for online comments.

Classify the comment into these six labels. Each label is independent (multi-label).
Return ONLY valid JSON with integer values 0 or 1 for each label.

Labels:
{label_block}

Required JSON format:
{json_template}

Comment:
\"\"\"{comment}\"\"\"

JSON:"""


FEW_SHOT_PROMPT = """You are a toxicity classifier for online comments.

Classify the comment into these six labels. Each label is independent (multi-label).
Return ONLY valid JSON with integer values 0 or 1 for each label.

Labels:
{label_block}

Required JSON format:
{json_template}

Examples:

Example 1
Comment: \"You are an idiot and nobody likes you.\"
JSON: {{"toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0}}

Example 2
Comment: \"Thank you for the helpful explanation!\"
JSON: {{"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}}

Example 3
Comment: \"I will find you and hurt you.\"
JSON: {{"toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0}}

Now classify this comment:

Comment:
\"\"\"{comment}\"\"\"

JSON:"""


def build_zero_shot_prompt(comment: str) -> str:
    return ZERO_SHOT_PROMPT.format(
        label_block=_label_block(),
        json_template=_json_template(),
        comment=comment,
    )


def build_few_shot_prompt(comment: str) -> str:
    return FEW_SHOT_PROMPT.format(
        label_block=_label_block(),
        json_template=_json_template(),
        comment=comment,
    )


def _extract_json_blob(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _coerce_label_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if value >= 0.5 else 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes"}:
            return 1
        if lowered in {"0", "false", "no"}:
            return 0
    return 0


def parse_llm_output(text: str) -> dict[str, int]:
    blob = _extract_json_blob(text)
    if blob is None:
        return {label: 0 for label in LABELS}

    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        parsed = {}

    result: dict[str, int] = {}
    for label in LABELS:
        result[label] = _coerce_label_value(parsed.get(label, 0))
    return result


def predictions_dict_to_array(prediction: dict[str, int]) -> list[int]:
    return [prediction[label] for label in LABELS]


def parse_with_regex_fallback(text: str) -> dict[str, int]:
    result = parse_llm_output(text)
    if any(result.values()):
        return result

    for label in LABELS:
        match = re.search(rf'"{label}"\s*:\s*([01])', text)
        if match:
            result[label] = int(match.group(1))
    return result
