"""Save a successful search; retry book_flight if the booking call fails."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any

from langchain_core.tools import StructuredTool

DEFAULT_RETRY_BUDGET = 2


@dataclass
class SearchCheckpoint:
    flights: list[dict[str, Any]] | None = None
    last_search_args: dict[str, Any] | None = None
    book_retries: int = 0


def apply_checkpoint(
    tools: list[Any],
    *,
    retry_budget: int = DEFAULT_RETRY_BUDGET,
    store: SearchCheckpoint | None = None,
) -> list[StructuredTool]:
    """Wrap search (snapshot) and book (retry). Same tool names as the input list."""
    store = store if store is not None else SearchCheckpoint()
    wrapped: list[StructuredTool] = []
    for tool in tools:
        if tool.name == "search_flights":
            wrapped.append(_wrap_search(tool, store))
        elif tool.name == "book_flight":
            wrapped.append(_wrap_book(tool, store, retry_budget))
        else:
            wrapped.append(tool)
    return wrapped


def _wrap_search(tool: Any, store: SearchCheckpoint) -> StructuredTool:
    inner = tool.func

    @wraps(inner)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = inner(*args, **kwargs)
        if _is_flight_list(result):
            store.flights = list(result)
            store.last_search_args = _search_args(args, kwargs)
        return result

    return _replace_func(tool, wrapped)


def _wrap_book(tool: Any, store: SearchCheckpoint, retry_budget: int) -> StructuredTool:
    inner = tool.func

    @wraps(inner)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = inner(*args, **kwargs)
        attempts = 0
        while not _book_ok(result) and attempts < retry_budget:
            attempts += 1
            store.book_retries = attempts
            result = inner(*args, **kwargs)
        return result

    return _replace_func(tool, wrapped)


def _replace_func(tool: Any, func: Any) -> StructuredTool:
    return StructuredTool.from_function(
        func=func,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def _is_flight_list(result: Any) -> bool:
    return isinstance(result, list) and all(isinstance(item, dict) and "id" in item for item in result)


def _book_ok(result: Any) -> bool:
    return isinstance(result, dict) and bool(result.get("ok"))


def _search_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if kwargs:
        return dict(kwargs)
    names = ("origin", "destination", "date")
    return {name: args[i] for i, name in enumerate(names) if i < len(args)}


def _demo() -> None:
    from agent.store import AirlineStore
    from agent.tools import make_booking_tools

    store = SearchCheckpoint()
    tools = apply_checkpoint(make_booking_tools(AirlineStore.open_trial()), store=store)
    search = next(t for t in tools if t.name == "search_flights")
    book = next(t for t in tools if t.name == "book_flight")
    search.invoke({"origin": "SFO", "destination": "JFK", "date": "2026-10-15"})
    booked = book.invoke({"flight_id": "UA100", "passenger_name": "Alice Chen"})
    print("snapshot ids:", [f["id"] for f in store.flights or []])
    print("search args:", store.last_search_args)
    print("book:", booked)
    print("retries:", store.book_retries)


if __name__ == "__main__":
    _demo()
