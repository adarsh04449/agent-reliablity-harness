"""Robustness: per-task mean of min(Acc_pert / Acc_0, 1), then mean across tasks."""

from __future__ import annotations

from typing import Any, Iterable

from metrics.util import filter_rows, pass_rate

PERTURBATIONS = ("paraphrase", "latency", "tool_failure")


def robustness(rows: Iterable[Any], *, mitigation: bool | None = False) -> float | None:
    rows = filter_rows(list(rows), mitigation=mitigation)
    task_ids = sorted({row.task_id for row in rows})
    scores: list[float] = []
    for task_id in task_ids:
        score = _robustness_one(filter_rows(rows, task_id=task_id))
        if score is not None:
            scores.append(score)
    if not scores:
        return None
    return sum(scores) / len(scores)


def _robustness_one(rows: list[Any]) -> float | None:
    acc_0 = pass_rate(filter_rows(rows, condition="baseline"))
    if acc_0 is None or acc_0 == 0:
        return None
    ratios: list[float] = []
    for condition in PERTURBATIONS:
        acc = pass_rate(filter_rows(rows, condition=condition))
        if acc is None:
            continue
        ratios.append(min(acc / acc_0, 1.0))
    if not ratios:
        return None
    return sum(ratios) / len(ratios)
