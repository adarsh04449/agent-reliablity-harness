"""Per-trial SQLite airline world. Seed is shared; each trial gets a memory clone."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.catalog import REPO_ROOT, load_task

STORE_DIR = REPO_ROOT / "store"
SCHEMA_PATH = STORE_DIR / "schema.sql"
SEED_PATH = STORE_DIR / "seed.sql"


@dataclass
class AirlineStore:
    conn: sqlite3.Connection
    passenger_name: str
    origin: str
    destination: str
    date: str
    time_of_day: str
    max_price: int
    gold_flight_id: str
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def open_trial(cls, task: dict[str, Any] | None = None) -> AirlineStore:
        data = task if task is not None else load_task()
        constraints = data.get("constraints") or {}
        # ToolNode runs tools on a worker thread; the runner may open the DB on another.
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.executescript(SEED_PATH.read_text(encoding="utf-8"))
        return cls(
            conn=conn,
            passenger_name=str(data["passenger_name"]),
            origin=str(data["origin"]).strip().upper(),
            destination=str(data["destination"]).strip().upper(),
            date=str(data["date"]),
            time_of_day=str(constraints.get("time_of_day", "morning")),
            max_price=int(constraints.get("max_price", 400)),
            gold_flight_id=str(data["gold_flight_id"]),
        )

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def search(self, origin: str, destination: str, date: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT id, origin, destination, date, depart, time_of_day, price, seats_left
                FROM flights
                WHERE origin = ? AND destination = ? AND date = ?
                ORDER BY depart, id
                """,
                (origin.strip().upper(), destination.strip().upper(), str(date)),
            ).fetchall()
            return [dict(row) for row in rows]

    def book(self, flight_id: str, passenger_name: str) -> dict[str, Any]:
        with self._lock:
            try:
                self.conn.execute("BEGIN")
                match = self.conn.execute(
                    "SELECT id, price, seats_left FROM flights WHERE id = ?",
                    (flight_id,),
                ).fetchone()
                if match is None:
                    self.conn.execute("ROLLBACK")
                    return {
                        "ok": False,
                        "error": "unknown_flight_id",
                        "flight_id": flight_id,
                    }
                if int(match["seats_left"]) < 1:
                    self.conn.execute("ROLLBACK")
                    return {"ok": False, "error": "sold_out", "flight_id": flight_id}
                already = self.conn.execute(
                    """
                    SELECT 1 FROM reservations
                    WHERE passenger_name = ? AND status = 'confirmed'
                    """,
                    (passenger_name,),
                ).fetchone()
                if already is not None:
                    self.conn.execute("ROLLBACK")
                    return {"ok": False, "error": "already_booked", "flight_id": flight_id}
                cursor = self.conn.execute(
                    """
                    INSERT INTO reservations (flight_id, passenger_name, status)
                    VALUES (?, ?, 'confirmed')
                    """,
                    (flight_id, passenger_name),
                )
                self.conn.execute(
                    "UPDATE flights SET seats_left = seats_left - 1 WHERE id = ?",
                    (flight_id,),
                )
                self.conn.execute("COMMIT")
                return {
                    "ok": True,
                    "confirmation_id": f"CONF-{cursor.lastrowid}",
                    "flight_id": flight_id,
                    "passenger_name": passenger_name,
                    "price": int(match["price"]),
                }
            except sqlite3.Error:
                self.conn.execute("ROLLBACK")
                raise

    def is_success(self, flight_id: str | None = None) -> bool:
        return self.outcome(flight_id) == "success"

    def outcome(self, flight_id: str | None = None) -> str:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT r.flight_id, f.origin, f.destination, f.date, f.time_of_day, f.price
                FROM reservations r
                JOIN flights f ON f.id = r.flight_id
                WHERE r.passenger_name = ? AND r.status = 'confirmed'
                ORDER BY r.id
                """,
                (self.passenger_name,),
            ).fetchall()
            if not rows:
                if flight_id is None:
                    return "no_booking"
                exists = self.conn.execute(
                    "SELECT 1 FROM flights WHERE id = ?",
                    (flight_id,),
                ).fetchone()
                return "unknown_flight" if exists is None else "no_booking"
            if len(rows) == 1 and self._matches_task(rows[0]):
                return "success"
            return "wrong_booking"

    def _matches_task(self, flight: sqlite3.Row) -> bool:
        return (
            str(flight["origin"]) == self.origin
            and str(flight["destination"]) == self.destination
            and str(flight["date"]) == self.date
            and str(flight["time_of_day"]) == self.time_of_day
            and int(flight["price"]) <= self.max_price
        )

    def counts(self) -> dict[str, int]:
        with self._lock:
            flights = int(self.conn.execute("SELECT COUNT(*) FROM flights").fetchone()[0])
            reservations = int(
                self.conn.execute("SELECT COUNT(*) FROM reservations").fetchone()[0]
            )
            airports = int(self.conn.execute("SELECT COUNT(*) FROM airports").fetchone()[0])
            sold_out = int(
                self.conn.execute(
                    "SELECT COUNT(*) FROM flights WHERE seats_left = 0"
                ).fetchone()[0]
            )
            return {
                "airports": airports,
                "flights": flights,
                "reservations": reservations,
                "sold_out": sold_out,
            }


def _demo() -> None:
    task = load_task()
    store = AirlineStore.open_trial(task)
    print("counts:", store.counts())
    results = store.search(task["origin"], task["destination"], task["date"])
    print("search (task date):", [row["id"] for row in results])
    print("search (wrong date):", [row["id"] for row in store.search("SFO", "JFK", "2026-10-16")])
    print("search (LAX):", [row["id"] for row in store.search("SFO", "LAX", task["date"])])
    gold = store.book(task["gold_flight_id"], task["passenger_name"])
    print("gold book:", gold, "outcome:", store.outcome())
    extra = store.book("DL1197", task["passenger_name"])
    print("second book:", extra, "outcome still:", store.outcome())
    other = AirlineStore.open_trial(task)
    cheap_morning = other.book("DL1197", task["passenger_name"])
    print("constraint-ok book:", cheap_morning, "outcome:", other.outcome())
    evening = AirlineStore.open_trial(task)
    distractor = evening.book("UA900", task["passenger_name"])
    print("evening book:", distractor, "outcome:", evening.outcome())
    print("is_success gold:", store.is_success())
    print("is_success DL1197:", other.is_success())
    print("is_success UA900:", evening.is_success())
    store.close()
    other.close()
    evening.close()


if __name__ == "__main__":
    _demo()
