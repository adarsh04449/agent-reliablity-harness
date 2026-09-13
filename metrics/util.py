"""Shared trial filters for the four scores."""

from __future__ import annotations

from typing import Any, Iterable


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
