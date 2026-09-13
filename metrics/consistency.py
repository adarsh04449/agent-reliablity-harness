"""Outcome consistency C_out on baseline repeats."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from metrics.util import filter_rows


def outcome_consistency(rows: Iterable[Any], *, mitigation: bool | None = False) -> float | None:
    baseline = filter_rows(rows, condition="baseline", mitigation=mitigation)
    if not baseline:
        return None
    by_task: dict[str, list[bool]] = defaultdict(list)
    for row in baseline:
        by_task[row.task_id].append(bool(row.success))
    scores = []
    for outcomes in by_task.values():
        p_hat = sum(outcomes) / len(outcomes)
        scores.append((2 * p_hat - 1) ** 2)
    return sum(scores) / len(scores)
