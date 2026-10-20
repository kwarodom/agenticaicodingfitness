"""
Exercise 08 — LangGraph Support Escalation Graph (Intermediate).

A stateful support graph with three tiers: L1 -> L2 -> Emergency. A triage
node classifies severity; conditional edges route accordingly. L1 can loop
back to itself on failure. Demonstrates cycles and multi-field state
mutation.

Key concepts:
    - Multi-field TypedDict state (severity, resolved, escalation_count)
    - Cycles — a node can route back to a previous node
    - Conditional routing based on multiple state fields
    - Functional node decomposition for each tier
"""

import operator
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")


class SupportState(TypedDict):
    issue: str
    severity: str
    messages: Annotated[list, operator.add]
    resolved: bool
    escalations: int


def triage(state: SupportState) -> SupportState:
    r = llm.invoke(
        "Classify severity as low/medium/high/critical.\n"
        f"Issue: {state['issue']}\n"
        "Reply with ONE word."
    )
    sev = r.content.strip().lower()
    print(f"[Triage] Severity: {sev}")
    return {
        "severity": sev,
        "messages": [AIMessage(content=f"Triaged: {sev}")],
    }


def l1_support(state: SupportState) -> SupportState:
    r = llm.invoke(
        f"You are hotel L1 support. Resolve this {state['severity']} issue:\n"
        f"{state['issue']}\nEnd with RESOLVED: YES or NO."
    )
    ok = "resolved: yes" in r.content.lower()
    print(f"[L1] Resolved: {ok}")
    return {
        "messages": [AIMessage(content=r.content)],
        "resolved": ok,
        "escalations": state["escalations"] + 1,
    }


def l2_expert(state: SupportState) -> SupportState:
    r = llm.invoke(
        f"You are a hotel systems engineer. HIGH priority:\n"
        f"{state['issue']}\nProvide technical fix. End RESOLVED: YES or NO."
    )
    ok = "resolved: yes" in r.content.lower()
    print(f"[L2 Expert] Resolved: {ok}")
    return {
        "messages": [AIMessage(content=r.content)],
        "resolved": ok,
        "escalations": state["escalations"] + 1,
    }


def emergency(state: SupportState) -> SupportState:
    r = llm.invoke(
        f"EMERGENCY PROTOCOL. Critical hotel issue:\n{state['issue']}\n"
        "Give immediate safety steps and notify authorities."
    )
    print("[EMERGENCY] Critical protocol activated!")
    return {
        "messages": [AIMessage(content=r.content)],
        "resolved": True,
    }


def route_triage(state: SupportState) -> str:
    s = state["severity"]
    if any(k in s for k in ("low", "medium")):
        return "l1"
    if "high" in s:
        return "l2"
    return "emergency"


def route_l1(state: SupportState) -> str:
    if state["resolved"] or state["escalations"] >= 2:
        return "end"
    return "l2"


builder = StateGraph(SupportState)
builder.add_node("triage", triage)
builder.add_node("l1", l1_support)
builder.add_node("l2", l2_expert)
builder.add_node("emergency", emergency)
builder.add_edge(START, "triage")
builder.add_conditional_edges(
    "triage",
    route_triage,
    {"l1": "l1", "l2": "l2", "emergency": "emergency"},
)
builder.add_conditional_edges("l1", route_l1, {"l2": "l2", "end": END})
builder.add_edge("l2", END)
builder.add_edge("emergency", END)
graph = builder.compile()


ISSUES = [
    "Room 302 AC is making a strange noise.",
    "3rd floor HVAC offline for 2 hours, guests complaining.",
    "Fire alarm triggered in kitchen — guests evacuating!",
]


if __name__ == "__main__":
    for issue in ISSUES:
        print(f"\n{'=' * 55}\nISSUE: {issue}")
        res = graph.invoke({
            "issue": issue,
            "severity": "",
            "messages": [HumanMessage(content=issue)],
            "resolved": False,
            "escalations": 0,
        })
        print(f"Resolved: {res['resolved']}")
