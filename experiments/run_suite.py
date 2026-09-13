"""Run real OpenAI trials, write logs, print the four reliability scores."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from harness.config import CONDITIONS, load_config
from harness.runner import TrialResult, run_suite
from metrics.report import score_table, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent reliability suite (OpenAI).")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="4 conditions × checkpoint on/off")
    parser.add_argument("--quiet", action="store_true", help="Hide agent think/tool trace")
    parser.add_argument("--task", dest="task_id", default=None, help="Run one task_id (default: all)")
    args = parser.parse_args()
    cfg = load_config(k=args.k, concurrency=args.concurrency)
    conditions = CONDITIONS if args.full else ("baseline",)
    mitigations = (False, True) if args.full else (False,)
    rows = asyncio.run(
        run_suite(
            cfg,
            conditions=conditions,
            mitigations=mitigations,
            verbose=not args.quiet,
            task_id=args.task_id,
        )
    )
    suite_id = _suite_id(rows)
    report_csv, plot_path = write_report(rows, suite_id=suite_id)
    table = score_table(rows)
    for group, scores in table.items():
        print(group)
        for name, value in scores.items():
            print(f"  {name}: {value if value is not None else 'n/a'}")
    print(report_csv)
    print(plot_path)


def _suite_id(rows: list[TrialResult]) -> str:
    if rows and rows[0].trace_path:
        return Path(rows[0].trace_path).stem
    return "suite"


if __name__ == "__main__":
    main()
