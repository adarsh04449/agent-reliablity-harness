"""Write one suite to results/logs/*.jsonl and results/csv/*.csv."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from agent.catalog import REPO_ROOT

LOG_DIR = REPO_ROOT / "results" / "logs"
CSV_DIR = REPO_ROOT / "results" / "csv"


def write_suite(results: list[Any], *, suite_id: str | None = None) -> tuple[Path, Path]:
    if not results:
        raise ValueError("write_suite needs at least one trial")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    name = suite_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jsonl_path = LOG_DIR / f"{name}.jsonl"
    csv_path = CSV_DIR / f"{name}.csv"
    jsonl_str = str(jsonl_path)
    for row in results:
        row.trace_path = jsonl_str
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(_row_dict(row), default=str) + "\n")
    frame = pd.DataFrame(
        [
            {
                **{key: value for key, value in _row_dict(row).items() if key != "tool_events"},
                "actions": " ".join(getattr(row, "actions", [])),
                "tool_events": json.dumps(getattr(row, "tool_events", [])),
            }
            for row in results
        ]
    )
    frame.to_csv(csv_path, index=False)
    return jsonl_path, csv_path


def _row_dict(row: Any) -> dict[str, Any]:
    if is_dataclass(row) and not isinstance(row, type):
        return asdict(row)
    return dict(row)
