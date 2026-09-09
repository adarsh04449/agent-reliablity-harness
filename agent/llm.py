"""OpenAI chat model for the booking agent."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "gpt-4o-mini"


def get_chat_model(*, model: str | None = None) -> ChatOpenAI:
    """gpt-4o-mini at temperature 0. Requires OPENAI_API_KEY."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required. Copy .env.example to .env and set your key."
        )
    return ChatOpenAI(
        model=model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        temperature=0,
        api_key=api_key,
    )
