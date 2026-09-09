"""Simulated booking tools for LangGraph. Catalog is fake; the LLM is real."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

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


@tool("search_flights")
def search_flights_tool(origin: str, destination: str, date: str) -> list[dict[str, Any]]:
    """Search the airline catalog for one-way flights. Date must be YYYY-MM-DD."""
    return search_flights(origin, destination, date)


@tool("book_flight")
def book_flight_tool(flight_id: str, passenger_name: str) -> dict[str, Any]:
    """Book a flight id from search results for the given passenger."""
    return book_flight(flight_id, passenger_name)


BOOKING_TOOLS = [search_flights_tool, book_flight_tool]
