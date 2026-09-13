"""Simulated booking tools. Each trial binds tools to its own SQLite clone."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from agent.store import AirlineStore


def make_booking_tools(store: AirlineStore) -> list[StructuredTool]:
    def search_flights(origin: str, destination: str, date: str) -> list[dict[str, Any]]:
        return store.search(origin, destination, date)

    def book_flight(flight_id: str, passenger_name: str) -> dict[str, Any]:
        return store.book(flight_id, passenger_name)

    return [
        StructuredTool.from_function(
            func=search_flights,
            name="search_flights",
            description="Search the airline catalog for one-way flights. Date must be YYYY-MM-DD.",
        ),
        StructuredTool.from_function(
            func=book_flight,
            name="book_flight",
            description="Book a flight id from search results for the given passenger.",
        ),
    ]
