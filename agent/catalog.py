"""Airline tool backend: in-memory catalog now, τ-bench-shaped later."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASK_PATH = REPO_ROOT / "tasks" / "flight_booking.yaml"


class FlightBackend(Protocol):
    """Minimal interface so a later τ-bench adapter can replace the catalog."""

    def search(self, origin: str, destination: str, date: str) -> list[dict[str, Any]]: ...

    def book(self, flight_id: str, passenger_name: str) -> dict[str, Any]: ...

    def is_success(self, flight_id: str | None) -> bool: ...


def load_task(path: Path | None = None) -> dict[str, Any]:
    task_path = path or DEFAULT_TASK_PATH
    with task_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass
class InMemoryAirline:
    flights: list[dict[str, Any]]
    gold_flight_id: str

    @classmethod
    def from_task(cls, task: dict[str, Any] | None = None) -> InMemoryAirline:
        data = task if task is not None else load_task()
        return cls(flights=list(data["flights"]), gold_flight_id=data["gold_flight_id"])

    def search(self, origin: str, destination: str, date: str) -> list[dict[str, Any]]:
        origin_n = origin.strip().upper()
        dest_n = destination.strip().upper()
        return [
            dict(flight)
            for flight in self.flights
            if flight["origin"] == origin_n
            and flight["destination"] == dest_n
            and str(flight["date"]) == str(date)
        ]

    def book(self, flight_id: str, passenger_name: str) -> dict[str, Any]:
        match = next((f for f in self.flights if f["id"] == flight_id), None)
        if match is None:
            return {
                "ok": False,
                "error": "unknown_flight_id",
                "flight_id": flight_id,
            }
        return {
            "ok": True,
            "confirmation_id": f"CONF-{flight_id}",
            "flight_id": flight_id,
            "passenger_name": passenger_name,
            "price": match["price"],
        }

    def is_success(self, flight_id: str | None) -> bool:
        return flight_id == self.gold_flight_id

    def outcome(self, flight_id: str | None) -> str:
        if flight_id is None:
            return "no_booking"
        if flight_id == self.gold_flight_id:
            return "success"
        if any(f["id"] == flight_id for f in self.flights):
            return "wrong_booking"
        return "unknown_flight"


def _demo() -> None:
    airline = InMemoryAirline.from_task()
    task = load_task()
    results = airline.search(task["origin"], task["destination"], task["date"])
    print("search (task date):", [f["id"] for f in results])
    print("search (wrong date):", [f["id"] for f in airline.search("SFO", "JFK", "2026-10-16")])
    print("search (LAX):", [f["id"] for f in airline.search("SFO", "LAX", task["date"])])
    gold = airline.book(task["gold_flight_id"], task["passenger_name"])
    distractor = airline.book("UA900", task["passenger_name"])
    ambiguous = airline.book("DL220", task["passenger_name"])
    print("gold book:", gold, "outcome:", airline.outcome(gold.get("flight_id")))
    print("distractor book:", distractor, "outcome:", airline.outcome(distractor.get("flight_id")))
    print("ambiguous morning book:", ambiguous, "outcome:", airline.outcome(ambiguous.get("flight_id")))
    print("is_success gold:", airline.is_success(task["gold_flight_id"]))
    print("is_success UA900:", airline.is_success("UA900"))
    print("is_success DL220:", airline.is_success("DL220"))


if __name__ == "__main__":
    _demo()
