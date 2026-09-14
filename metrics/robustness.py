"""Robustness: per-task min(Acc_pert / Acc_0, 1), then mean across tasks."""

from __future__ import annotations

from typing import Any, Iterable

from metrics.util import filter_rows, mean_per_task, pass_rate

PERTURBATIONS = ("paraphrase", "latency", "tool_failure")
SPLIT_NAMES = ("prompt", "latency", "fault")


def robustness(rows: Iterable[Any], *, mitigation: bool | None = False) -> float | None:
    parts = [robustness_split(rows, mitigation=mitigation).get(key) for key in ("prompt", "latency", "fault")]
    values = [part for part in parts if part is not None]
    if not values:
        return None
    return sum(values) / len(values)


def robustness_split(
    rows: Iterable[Any], *, mitigation: bool | None = False
) -> dict[str, float | None]:
    rows = list(rows)
    return {
        name: mean_per_task(
            rows,
            lambda sliced, cond=condition: _ratio(sliced, cond),
            mitigation=mitigation,
        )
        for condition, name in zip(PERTURBATIONS, SPLIT_NAMES)
    }


def _ratio(rows: list[Any], condition: str) -> float | None:
    acc_0 = pass_rate(filter_rows(rows, condition="baseline"))
    acc = pass_rate(filter_rows(rows, condition=condition))
    if acc_0 is None or acc_0 == 0 or acc is None:
        return None
    return min(acc / acc_0, 1.0)
