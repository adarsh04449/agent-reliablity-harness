"""Bounded severity: 1 - mean harm weight on failures."""

from __future__ import annotations

from typing import Any, Iterable


def bounded_severity(rows: Iterable[Any]) -> float | None:
    failures = [row for row in rows if not row.success]
    if not failures:
        return None
    mean_w = sum(float(row.severity) for row in failures) / len(failures)
    return 1.0 - mean_w
