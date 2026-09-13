"""Suite settings from .env (CLI overrides come later in run_suite)."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Condition = Literal["baseline", "paraphrase", "latency", "tool_failure"]
CONDITIONS: tuple[Condition, ...] = ("baseline", "paraphrase", "latency", "tool_failure")


class HarnessConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    k: int = 2
    concurrency: int = 2
    p_fault: float = 0.2
    seed: int = 0
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    @property
    def model(self) -> str:
        return self.openai_model


def load_config(**overrides: object) -> HarnessConfig:
    cfg = HarnessConfig()
    if not overrides:
        return cfg
    return cfg.model_copy(update={key: value for key, value in overrides.items() if value is not None})


def fault_rng_seed(task_id: str, condition: str, repeat: int, *, base_seed: int = 0) -> int:
    """Stable seed so checkpoint on/off see the same injected faults."""
    payload = f"{base_seed}:{task_id}:{condition}:{repeat}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _demo() -> None:
    cfg = load_config()
    print(cfg.model_dump())
    print("model:", cfg.model)
    print("fault seed:", fault_rng_seed("flight_booking_sfo_jfk", "tool_failure", 0, base_seed=cfg.seed))


if __name__ == "__main__":
    _demo()
