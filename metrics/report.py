"""Four core scores → CSV + bar chart of pass rate by condition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from agent.catalog import REPO_ROOT
from metrics.consistency import outcome_consistency
from metrics.predictability import predictability
from metrics.robustness import robustness
from metrics.severity import bounded_severity
from metrics.util import filter_rows, pass_rate

PLOT_DIR = REPO_ROOT / "results" / "plots"
CSV_DIR = REPO_ROOT / "results" / "csv"


def score_table(rows: list[Any]) -> dict[str, dict[str, float | None]]:
    table: dict[str, dict[str, float | None]] = {}
    for mitigation in (False, True):
        sliced = filter_rows(rows, mitigation=mitigation)
        if not sliced:
            continue
        pred = predictability(sliced)
        key = "checkpoint" if mitigation else "no_checkpoint"
        table[key] = {
            "consistency": outcome_consistency(rows, mitigation=mitigation),
            "robustness": robustness(rows, mitigation=mitigation),
            "predictability_brier": pred["brier"],
            "calibration": pred["calibration"],
            "auroc": pred["auroc"],
            "bounded_severity": bounded_severity(sliced),
            "pass_rate_baseline": pass_rate(filter_rows(sliced, condition="baseline")),
        }
    return table


def write_report(rows: list[Any], *, suite_id: str) -> tuple[Path, Path]:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    table = score_table(rows)
    frame = pd.DataFrame(table).T
    csv_path = CSV_DIR / f"{suite_id}_reliability.csv"
    frame.to_csv(csv_path)
    plot_path = PLOT_DIR / f"{suite_id}_pass_rate.png"
    _pass_rate_chart(rows, plot_path)
    return csv_path, plot_path


def _pass_rate_chart(rows: list[Any], path: Path) -> None:
    conditions = sorted({row.condition for row in rows})
    labels = []
    values = []
    for condition in conditions:
        for mitigation, name in ((False, "off"), (True, "on")):
            rate = pass_rate(filter_rows(rows, condition=condition, mitigation=mitigation))
            if rate is None:
                continue
            labels.append(f"{condition}\nckpt {name}")
            values.append(rate)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("pass rate")
    ax.set_title("Pass rate by condition")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
