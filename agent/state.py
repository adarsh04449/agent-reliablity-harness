"""LangGraph agent state. Graph and confidence node read/write these fields."""

from __future__ import annotations

from typing import Annotated, Any, NotRequired, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """ReAct loop plus post-run confidence. Checkpoints land here in a later step."""

    messages: Annotated[list[BaseMessage], add_messages]
    instruction: str
    confidence: NotRequired[float]
    booked_flight_id: NotRequired[str | None]
    last_book_ok: NotRequired[bool]
    success: NotRequired[bool]
    outcome: NotRequired[str]
    tool_events: NotRequired[list[dict[str, Any]]]
