"""Latency and partial-failure wrappers around booking tools. Applied per trial."""

from __future__ import annotations

import random
import time
from functools import wraps
from typing import Any

from langchain_core.tools import StructuredTool

from agent.store import AirlineStore
from agent.tools import make_booking_tools

DEFAULT_DELAY_S = 1.0
DEFAULT_P_FAULT = 0.2
_FAULT_KINDS = ("timeout", "empty", "malformed")


def wrap_booking_tools(
    *,
    tools: list[Any] | None = None,
    delay_s: float = 0.0,
    p_fault: float = 0.0,
    rng: random.Random | None = None,
    events: list[dict[str, Any]] | None = None,
) -> list[StructuredTool]:
    """Same tool names/schemas as the bound booking tools, with optional delay and faults."""
    rng = rng or random.Random()
    events = events if events is not None else []
    base = list(tools) if tools is not None else make_booking_tools(AirlineStore.open_trial())
    return [
        _wrap_one(tool, delay_s=delay_s, p_fault=p_fault, rng=rng, events=events)
        for tool in base
    ]


def _wrap_one(
    tool: Any,
    *,
    delay_s: float,
    p_fault: float,
    rng: random.Random,
    events: list[dict[str, Any]],
) -> StructuredTool:
    inner = tool.func

    @wraps(inner)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        if delay_s > 0:
            time.sleep(delay_s)
        injected = p_fault > 0 and rng.random() < p_fault
        if injected:
            kind = rng.choice(_FAULT_KINDS)
            result = _fault_result(tool.name, kind)
            ok = False
        else:
            result = inner(*args, **kwargs)
            ok = _call_ok(tool.name, result)
        events.append(
            {
                "tool_name": tool.name,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "fault_injected": injected,
                "ok": ok,
            }
        )
        return result

    return StructuredTool.from_function(
        func=wrapped,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def _fault_result(tool_name: str, kind: str) -> Any:
    if kind == "timeout":
        return {"ok": False, "error": "timeout"}
    if kind == "empty":
        return [] if tool_name == "search_flights" else {}
    return "<<<malformed>>>"


def _call_ok(tool_name: str, result: Any) -> bool:
    if tool_name == "book_flight" and isinstance(result, dict):
        return bool(result.get("ok"))
    return True


def _demo() -> None:
    events: list[dict[str, Any]] = []
    tools = wrap_booking_tools(delay_s=0.0, p_fault=1.0, rng=random.Random(0), events=events)
    search = next(t for t in tools if t.name == "search_flights")
    print("forced fault:", search.invoke({"origin": "SFO", "destination": "JFK", "date": "2026-10-15"}))
    print("events:", events)


if __name__ == "__main__":
    _demo()
