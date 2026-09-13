"""Run booking trials (one fresh graph each). Default: baseline only."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

from agent.catalog import load_task, load_tasks
from agent.graph import run_booking
from agent.store import AirlineStore
from harness.config import Condition, HarnessConfig, fault_rng_seed, load_config
from harness.logging import write_suite
from perturbation.paraphrase import paraphrase_at
from perturbation.wrappers import DEFAULT_DELAY_S

SEVERITY = {
    "success": 0.0,
    "wrong_booking": 1.0,
    "no_booking": 0.25,
    "unknown_flight": 0.25,
    "sold_out": 0.25,
}


@dataclass
class TrialResult:
    task_id: str
    condition: str
    repeat: int
    success: bool
    severity: float
    confidence: float
    actions: list[str]
    latency_ms: float
    tool_calls: int
    trace_path: str = ""
    mitigation: bool = False
    outcome: str = ""
    tool_events: list[dict[str, Any]] = field(default_factory=list)


def iter_jobs(
    cfg: HarnessConfig,
    tasks: list[dict[str, Any]],
    *,
    conditions: tuple[Condition, ...] = ("baseline",),
    mitigations: tuple[bool, ...] = (False,),
) -> list[tuple[dict[str, Any], Condition, int, bool]]:
    return [
        (task, condition, repeat, mitigation)
        for task in tasks
        for condition in conditions
        for repeat in range(cfg.k)
        for mitigation in mitigations
    ]


async def run_trial(
    cfg: HarnessConfig,
    *,
    condition: Condition,
    repeat: int,
    mitigation: bool,
    task: dict[str, Any] | None = None,
    verbose: bool = True,
) -> TrialResult:
    task = task if task is not None else load_task()
    events: list[dict[str, Any]] = []
    instruction = task["instruction"].strip()
    if condition == "paraphrase":
        instruction = paraphrase_at(task, repeat)
    delay_s = DEFAULT_DELAY_S if condition == "latency" else 0.0
    p_fault = cfg.p_fault if condition == "tool_failure" else 0.0
    rng = random.Random(
        fault_rng_seed(task["task_id"], condition, repeat, base_seed=cfg.seed)
    )
    started = time.perf_counter()
    if verbose:
        print(
            f"\n=== trial {task['task_id']} {condition} "
            f"repeat={repeat} checkpoint={mitigation} ==="
        )
    store = AirlineStore.open_trial(task)
    try:
        state = await asyncio.to_thread(
            run_booking,
            instruction,
            model=cfg.model,
            store=store,
            delay_s=delay_s,
            p_fault=p_fault,
            rng=rng,
            events=events,
            checkpoint=mitigation,
            verbose=verbose,
        )
    finally:
        store.close()
    wall_ms = (time.perf_counter() - started) * 1000
    outcome = str(state.get("outcome") or "no_booking")
    return TrialResult(
        task_id=task["task_id"],
        condition=condition,
        repeat=repeat,
        success=bool(state.get("success")),
        severity=SEVERITY.get(outcome, 0.25),
        confidence=float(state.get("confidence") or 0.5),
        actions=[event["tool_name"] for event in events],
        latency_ms=wall_ms,
        tool_calls=len(events),
        mitigation=mitigation,
        outcome=outcome,
        tool_events=list(events),
    )


async def run_suite(
    cfg: HarnessConfig | None = None,
    *,
    conditions: tuple[Condition, ...] = ("baseline",),
    mitigations: tuple[bool, ...] = (False,),
    verbose: bool = True,
    task_id: str | None = None,
) -> list[TrialResult]:
    cfg = cfg if cfg is not None else load_config()
    tasks = load_tasks(task_id=task_id)
    semaphore = asyncio.Semaphore(cfg.concurrency)

    async def bound(job: tuple[dict[str, Any], Condition, int, bool]) -> TrialResult:
        task, condition, repeat, mitigation = job
        async with semaphore:
            return await run_trial(
                cfg,
                condition=condition,
                repeat=repeat,
                mitigation=mitigation,
                task=task,
                verbose=verbose,
            )

    rows = list(
        await asyncio.gather(
            *[
                bound(job)
                for job in iter_jobs(
                    cfg, tasks, conditions=conditions, mitigations=mitigations
                )
            ]
        )
    )
    jsonl_path, csv_path = write_suite(rows)
    print(jsonl_path)
    print(csv_path)
    return rows


def _print_results(results: list[TrialResult]) -> None:
    for row in results:
        print(
            row.condition,
            "rep",
            row.repeat,
            "mitigation",
            row.mitigation,
            "success",
            row.success,
            "outcome",
            row.outcome,
            "confidence",
            row.confidence,
        )


if __name__ == "__main__":
    _print_results(asyncio.run(run_suite()))
