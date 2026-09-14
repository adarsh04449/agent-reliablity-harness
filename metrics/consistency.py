"""Outcome consistency plus stretch trajectory / resource scores on baseline repeats."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np

from metrics.util import mean_per_task


def outcome_consistency(rows: Iterable[Any], *, mitigation: bool | None = False) -> float | None:
    return mean_per_task(rows, _c_out, mitigation=mitigation, condition="baseline")


def trajectory_mix(rows: Iterable[Any], *, mitigation: bool | None = False) -> float | None:
    """1 - mean pairwise Jensen–Shannon divergence of tool-name bags (baseline)."""
    return mean_per_task(rows, _mix, mitigation=mitigation, condition="baseline")


def trajectory_order(rows: Iterable[Any], *, mitigation: bool | None = False) -> float | None:
    """Mean pairwise 1 - normalized Levenshtein distance on tool sequences (baseline)."""
    return mean_per_task(rows, _order, mitigation=mitigation, condition="baseline")


def resource_stability(rows: Iterable[Any], *, mitigation: bool | None = False) -> float | None:
    """1 / (1 + CV) of tool-call counts on baseline repeats."""
    return mean_per_task(rows, _resource, mitigation=mitigation, condition="baseline")


def _c_out(rows: list[Any]) -> float | None:
    if not rows:
        return None
    p_hat = sum(bool(row.success) for row in rows) / len(rows)
    return (2 * p_hat - 1) ** 2


def _mix(rows: list[Any]) -> float | None:
    bags = [_action_dist(list(row.actions or [])) for row in rows]
    return _mean_pairwise(bags, lambda a, b: 1.0 - _jsd(a, b))


def _order(rows: list[Any]) -> float | None:
    seqs = [list(row.actions or []) for row in rows]
    return _mean_pairwise(seqs, _normalized_similarity)


def _resource(rows: list[Any]) -> float | None:
    if len(rows) < 2:
        return None
    counts = np.array([float(row.tool_calls) for row in rows], dtype=float)
    mean = float(counts.mean())
    if mean <= 0:
        return None
    cv = float(counts.std(ddof=0) / mean)
    return 1.0 / (1.0 + cv)


def _mean_pairwise(items: list[Any], score_fn) -> float | None:
    if len(items) < 2:
        return None
    scores = [
        score_fn(items[i], items[j])
        for i in range(len(items))
        for j in range(i + 1, len(items))
    ]
    return sum(scores) / len(scores)


def _action_dist(actions: list[str]) -> dict[str, float]:
    if not actions:
        return {}
    counts = Counter(actions)
    total = sum(counts.values())
    return {name: n / total for name, n in counts.items()}


def _jsd(p: dict[str, float], q: dict[str, float]) -> float:
    keys = sorted(set(p) | set(q))
    if not keys:
        return 0.0
    p_vec = np.array([p.get(k, 0.0) for k in keys])
    q_vec = np.array([q.get(k, 0.0) for k in keys])
    m = 0.5 * (p_vec + q_vec)
    return 0.5 * _kl(p_vec, m) + 0.5 * _kl(q_vec, m)


def _kl(p: np.ndarray, m: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / m[mask])))


def _normalized_similarity(a: list[str], b: list[str]) -> float:
    denom = max(len(a), len(b))
    if denom == 0:
        return 1.0
    return 1.0 - _levenshtein(a, b) / denom


def _levenshtein(a: list[str], b: list[str]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        cur = [i]
        for j, token_b in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (token_a != token_b)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]
