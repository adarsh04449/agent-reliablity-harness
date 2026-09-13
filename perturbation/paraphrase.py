"""Hand-written rewordings from each task YAML. No extra LLM call."""

from __future__ import annotations

from typing import Any


def paraphrases_for(task: dict[str, Any]) -> list[str]:
    raw = task.get("paraphrases") or []
    lines = [str(item).strip() for item in raw if str(item).strip()]
    if lines:
        return lines
    return [str(task["instruction"]).strip()]


def paraphrase_at(task: dict[str, Any], index: int) -> str:
    """Cycle through this task's paraphrases."""
    options = paraphrases_for(task)
    return options[index % len(options)]
