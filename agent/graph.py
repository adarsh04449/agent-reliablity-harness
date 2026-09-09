"""ReAct booking graph: agent ↔ tools, then a post-run confidence question."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.catalog import load_task
from agent.llm import get_chat_model
from agent.state import AgentState
from agent.tools import BOOKING_TOOLS, get_airline

MAX_AGENT_STEPS = 8
CONFIDENCE_PROMPT = (
    "On a scale from 0 to 1, how confident are you that you completed "
    "the booking correctly? Reply with a single number."
)
SYSTEM_PROMPT = (
    "You book one-way flights using two tools: search_flights then book_flight. "
    "Search first. Pick a flight that matches the passenger constraints "
    "(morning departure, budget). Book using a flight id from search results "
    "and the passenger name in the request. Do not invent flight ids."
)
_CONFIDENCE_RE = re.compile(r"(?<![\d.])(0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])")


def parse_confidence(text: str) -> float:
    """First 0–1 float in the reply; 0.5 if none."""
    if not text:
        return 0.5
    match = _CONFIDENCE_RE.search(text)
    if match is None:
        return 0.5
    return float(match.group(1))


def build_graph(*, model: str | None = None):
    llm = get_chat_model(model=model)
    llm_with_tools = llm.bind_tools(BOOKING_TOOLS)

    def agent(state: AgentState) -> dict[str, Any]:
        messages = list(state["messages"])
        if not messages:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=state["instruction"]),
            ]
        return {"messages": [llm_with_tools.invoke(messages)]}

    def confidence(state: AgentState) -> dict[str, Any]:
        ask = list(state["messages"]) + [HumanMessage(content=CONFIDENCE_PROMPT)]
        reply = llm.invoke(ask)
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        booked_id, last_ok = _last_booking(state["messages"])
        airline = get_airline()
        return {
            "messages": [reply],
            "confidence": parse_confidence(text),
            "booked_flight_id": booked_id,
            "last_book_ok": last_ok,
            "success": airline.is_success(booked_id),
            "outcome": airline.outcome(booked_id),
        }

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(BOOKING_TOOLS))
    graph.add_node("confidence", confidence)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _route_agent, {"tools": "tools", "confidence": "confidence"})
    graph.add_edge("tools", "agent")
    graph.add_edge("confidence", END)
    return graph.compile()


def run_booking(instruction: str | None = None, *, model: str | None = None) -> AgentState:
    task = load_task()
    text = instruction if instruction is not None else task["instruction"].strip()
    graph = build_graph(model=model)
    return graph.invoke(
        {
            "instruction": text,
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=text),
            ],
            "tool_events": [],
        },
        config={"recursion_limit": MAX_AGENT_STEPS * 2 + 4},
    )


def _route_agent(state: AgentState) -> Literal["tools", "confidence"]:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    agent_steps = sum(1 for m in state["messages"] if isinstance(m, AIMessage))
    if tool_calls and agent_steps < MAX_AGENT_STEPS:
        return "tools"
    return "confidence"


def _last_booking(messages: list[BaseMessage]) -> tuple[str | None, bool]:
    booked_id: str | None = None
    last_ok = False
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name != "book_flight":
            continue
        payload = _as_dict(message.content)
        booked_id = payload.get("flight_id")
        last_ok = bool(payload.get("ok"))
    return booked_id, last_ok


def _as_dict(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _demo() -> None:
    result = run_booking()
    print("success:", result.get("success"))
    print("outcome:", result.get("outcome"))
    print("booked:", result.get("booked_flight_id"))
    print("confidence:", result.get("confidence"))
    print("last_book_ok:", result.get("last_book_ok"))


if __name__ == "__main__":
    _demo()
