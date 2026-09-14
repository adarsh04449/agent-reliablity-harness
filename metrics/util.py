"""Shared trial filters for the four scores."""

from __future__ import annotations

from typing import Any, Callable, Iterable


def filter_rows(
    rows: Iterable[Any],
    *,
    condition: str | None = None,
    mitigation: bool | None = None,
    task_id: str | None = None,
) -> list[Any]:
    out: list[Any] = []
    for row in rows:
        if condition is not None and row.condition != condition:
            continue
        if mitigation is not None and bool(row.mitigation) != mitigation:
            continue
        if task_id is not None and row.task_id != task_id:
            continue
        out.append(row)
    return out


def pass_rate(rows: Iterable[Any]) -> float | None:
    rows = list(rows)
    if not rows:
        return None
    return sum(1 for row in rows if row.success) / len(rows)


def mean_per_task(
    rows: Iterable[Any],
    score_fn: Callable[[list[Any]], float | None],
    *,
    mitigation: bool | None = False,
    condition: str | None = None,
) -> float | None:
    rows = filter_rows(list(rows), mitigation=mitigation, condition=condition)
    scores: list[float] = []
    for task_id in sorted({row.task_id for row in rows}):
        score = score_fn(filter_rows(rows, task_id=task_id))
        if score is not None:
            scores.append(score)
    if not scores:
        return None
    return sum(scores) / len(scores)
