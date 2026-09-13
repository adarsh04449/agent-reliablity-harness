"""Robustness: mean of min(Acc_pert / Acc_0, 1) over paraphrase, latency, fault."""

from __future__ import annotations

from typing import Any, Iterable

from metrics.util import filter_rows, pass_rate

PERTURBATIONS = ("paraphrase", "latency", "tool_failure")


def robustness(rows: Iterable[Any], *, mitigation: bool | None = False) -> float | None:
    rows = list(rows)
    acc_0 = pass_rate(filter_rows(rows, condition="baseline", mitigation=mitigation))
    if acc_0 is None or acc_0 == 0:
        return None
    ratios: list[float] = []
    for condition in PERTURBATIONS:
        acc = pass_rate(filter_rows(rows, condition=condition, mitigation=mitigation))
        if acc is None:
            continue
        ratios.append(min(acc / acc_0, 1.0))
    if not ratios:
        return None
    return sum(ratios) / len(ratios)
