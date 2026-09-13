"""Per-trial SQLite airline world. Seed is shared; each trial gets a memory clone."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.catalog import REPO_ROOT, load_task

STORE_DIR = REPO_ROOT / "store"
SCHEMA_PATH = STORE_DIR / "schema.sql"
SEED_PATH = STORE_DIR / "seed.sql"


@dataclass
class AirlineStore:
    conn: sqlite3.Connection
    gold_flight_id: str
    passenger_name: str

    @classmethod
    def open_trial(cls, task: dict[str, Any] | None = None) -> AirlineStore:
        data = task if task is not None else load_task()
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.executescript(SEED_PATH.read_text(encoding="utf-8"))
        return cls(
            conn=conn,
            gold_flight_id=str(data["gold_flight_id"]),
            passenger_name=str(data["passenger_name"]),
        )

    def close(self) -> None:
        self.conn.close()

    def search(self, origin: str, destination: str, date: str) -> list[dict[str, Any]]:
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
                WHERE flight_id = ? AND passenger_name = ? AND status = 'confirmed'
                """,
                (flight_id, passenger_name),
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
        rows = self.conn.execute(
            """
            SELECT flight_id FROM reservations
            WHERE passenger_name = ? AND status = 'confirmed'
            ORDER BY id
            """,
            (self.passenger_name,),
        ).fetchall()
        booked = [str(row["flight_id"]) for row in rows]
        if not booked:
            if flight_id is None:
                return "no_booking"
            exists = self.conn.execute(
                "SELECT 1 FROM flights WHERE id = ?",
                (flight_id,),
            ).fetchone()
            return "unknown_flight" if exists is None else "no_booking"
        if booked == [self.gold_flight_id]:
            return "success"
        return "wrong_booking"

    def counts(self) -> dict[str, int]:
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
    print("gold book:", gold, "outcome:", store.outcome(gold.get("flight_id")))
    other = AirlineStore.open_trial(task)
    distractor = other.book("UA900", task["passenger_name"])
    print("distractor book:", distractor, "outcome:", other.outcome(distractor.get("flight_id")))
    print("is_success gold:", store.is_success())
    print("is_success UA900:", other.is_success())
    store.close()
    other.close()


if __name__ == "__main__":
    _demo()
