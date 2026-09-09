"""Mock booking tools. No LLM — call these directly or wrap them in LangGraph later."""

from __future__ import annotations

from typing import Any

from agent.catalog import InMemoryAirline

_default_airline: InMemoryAirline | None = None


def get_airline() -> InMemoryAirline:
    global _default_airline
    if _default_airline is None:
        _default_airline = InMemoryAirline.from_task()
    return _default_airline


def search_flights(
    origin: str,
    destination: str,
    date: str,
    airline: InMemoryAirline | None = None,
) -> list[dict[str, Any]]:
    backend = airline or get_airline()
    return backend.search(origin, destination, date)


def book_flight(
    flight_id: str,
    passenger_name: str,
    airline: InMemoryAirline | None = None,
) -> dict[str, Any]:
    backend = airline or get_airline()
    return backend.book(flight_id, passenger_name)
