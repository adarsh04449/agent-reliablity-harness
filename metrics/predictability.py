"""Predictability from post-run confidence: Brier, calibration, AUROC."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


def predictability(rows: Iterable[Any]) -> dict[str, float | None]:
    rows = list(rows)
    if not rows:
        return {"brier": None, "calibration": None, "auroc": None}
    y = np.array([1.0 if row.success else 0.0 for row in rows])
    c = np.array([float(row.confidence) for row in rows])
    brier = 1.0 - float(np.mean((c - y) ** 2))
    calibration = float(1.0 - _ece(c, y))
    auroc: float | None
    if y.min() == y.max():
        auroc = None
    else:
        auroc = float(roc_auc_score(y, c))
    return {"brier": brier, "calibration": calibration, "auroc": auroc}


def _ece(confidence: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    n = len(y)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if not np.any(mask):
            continue
        acc = float(y[mask].mean())
        conf = float(confidence[mask].mean())
        total += (mask.sum() / n) * abs(acc - conf)
    return total
