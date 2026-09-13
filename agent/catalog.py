"""Task YAML loader. Flight data lives in the SQLite seed (store/)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "tasks"
DEFAULT_TASK_PATH = TASKS_DIR / "flight_booking.yaml"


class FlightBackend(Protocol):
    """search / book / is_success so the runner can swap backends later."""

    def search(self, origin: str, destination: str, date: str) -> list[dict[str, Any]]: ...

    def book(self, flight_id: str, passenger_name: str) -> dict[str, Any]: ...

    def is_success(self, flight_id: str | None = None) -> bool: ...


def load_task(path: Path | None = None) -> dict[str, Any]:
    task_path = path or DEFAULT_TASK_PATH
    with task_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_tasks(*, task_id: str | None = None) -> list[dict[str, Any]]:
    tasks = [
        load_task(path)
        for path in sorted(TASKS_DIR.glob("*.yaml"))
    ]
    if task_id is None:
        return tasks
    matched = [task for task in tasks if task["task_id"] == task_id]
    if not matched:
        known = ", ".join(task["task_id"] for task in tasks)
        raise ValueError(f"unknown task_id {task_id!r}; expected one of: {known}")
    return matched


def _demo() -> None:
    from agent.store import _demo as store_demo

    store_demo()


if __name__ == "__main__":
    _demo()
