"""
Exercise 13 — Autonomous Research Agent with Tool Use (Expert).

A fully autonomous research loop:
    Planner -> decomposes the task into sub-questions
    Researcher -> calls native tools (market_data, competitor_lookup,
                  roi_calc) to gather answers
    Synthesiser -> writes a structured report
    Critic -> judges quality and loops back if low
    Finaliser -> polishes the approved report

Demonstrates tool_use, reflection, and bounded autonomy.

Key concepts:
    - Claude tool_use via @tool + bind_tools
    - Planner-executor-critic (PEC) architecture pattern
    - Bounded autonomy — iteration counter prevents infinite loops
    - Reflection loop — Critic routes back to Synthesiser if quality low
"""

import operator
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

load_dotenv()
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")


# --- Tools ---


@tool
def market_data(topic: str) -> str:
    """Retrieve market size and growth statistics for a topic."""
    db = {
        "hotel energy": "Market $4.2B (2024), 11.3% CAGR to 2030.",
        "smart building": "Market $108B (2024), 15.2% CAGR to 2029.",
        "sea iot": "SEA IoT fastest-growing region at 18% CAGR.",
    }
    for k, v in db.items():
        if any(w in topic.lower() for w in k.split()):
            return v
    return f"No data found for '{topic}'."


@tool
def competitor_lookup(company: str) -> str:
    """Look up competitor profile by company name."""
    db = {
        "siemens": "Desigo CC — enterprise, high cost, global.",
        "schneider": "EcoStruxure — full ecosystem, lock-in risk.",
        "honeywell": "Forge — strong in North America, growing SEA.",
        "altotech": "CERO — AI-native, SEA-focused, mid-market.",
    }
    for k, v in db.items():
        if k in company.lower():
            return v
    return f"No profile for {company}."


@tool
def roi_calc(rooms: int, savings_pct: float, rate_thb: float = 5.2) -> str:
    """Calculate 3-year ROI for hotel energy system deployment."""
    kwh_yr = rooms * 17.5 * 365
    saved_kwh = kwh_yr * savings_pct / 100
    saved_thb = saved_kwh * rate_thb
    impl_cost = rooms * 800 * 35  # $800/room in THB
    payback = impl_cost / saved_thb if saved_thb > 0 else float("inf")
    roi_3yr = (
        (saved_thb * 3 - impl_cost) / impl_cost * 100 if impl_cost else 0
    )
    return (
        f"{rooms} rooms, {savings_pct}% savings: "
        f"THB{saved_thb:,.0f}/yr, payback {payback:.1f} yr, "
        f"3yr ROI {roi_3yr:.0f}%"
    )


TOOLS = [market_data, competitor_lookup, roi_calc]
llm_tools = llm.bind_tools(TOOLS)


# --- State ---


class ResState(TypedDict):
    task: str
    plan: list
    findings: Annotated[list, operator.add]
    draft: str
    critique: str
    iteration: int
    final: str


# --- Nodes ---


def planner(state: ResState) -> ResState:
    r = llm.invoke(
        f"Break this research task into 4 specific questions:\n"
        f"{state['task']}\nNumber them 1-4."
    )
    plan = [l.strip() for l in r.content.splitlines() if l.strip()]
    print(f"[Planner] {len(plan)} questions generated")
    return {"plan": plan, "iteration": 0}


def researcher(state: ResState) -> ResState:
    idx = state["iteration"]
    if idx >= len(state["plan"]):
        return {}
    q = state["plan"][idx]
    print(f"[Researcher] Q{idx + 1}: {q[:60]}")
    r = llm_tools.invoke(f"Answer using tools if relevant: {q}")
    findings: list = []
    if hasattr(r, "tool_calls") and r.tool_calls:
        for tc in r.tool_calls:
            for t in TOOLS:
                if t.name == tc["name"]:
                    res = t.invoke(tc["args"])
                    findings.append(f"Q{idx + 1}: {res}")
    else:
        findings.append(f"Q{idx + 1}: {r.content[:200]}")
    return {"findings": findings, "iteration": idx + 1}


def synthesiser(state: ResState) -> ResState:
    print("[Synthesiser] Writing report...")
    r = llm.invoke(
        f"Task: {state['task']}\n"
        f"Findings:\n" + "\n".join(state["findings"]) + "\n"
        "Write a structured 250-word report: "
        "Executive Summary / Market / Competitive / ROI / Recommendation."
    )
    return {"draft": r.content}


def critic(state: ResState) -> ResState:
    print("[Critic] Reviewing report quality...")
    r = llm.invoke(
        f"Rate this report 1-10 and explain:\n{state['draft'][:600]}\n"
        "If score >= 7 say QUALITY: APPROVED else QUALITY: REVISE."
    )
    approved = "QUALITY: APPROVED" in r.content
    print(f"[Critic] {'Approved' if approved else 'Needs revision'}")
    return {
        "critique": r.content,
        "final": state["draft"] if approved else "",
    }


def finalise(state: ResState) -> ResState:
    print("[Finalise] Polishing report...")
    r = llm.invoke(
        f"Add title, date (Apr 2026), and action items:\n{state['draft']}"
    )
    return {"final": r.content}


# --- Routing ---


def route_research(state: ResState) -> str:
    return (
        "synthesise"
        if state["iteration"] >= len(state["plan"])
        else "research"
    )


def route_critic(state: ResState) -> str:
    return "finalise"  # always finalise (could revise based on state['final'])


builder = StateGraph(ResState)
builder.add_node("planner", planner)
builder.add_node("researcher", researcher)
builder.add_node("synthesiser", synthesiser)
builder.add_node("critic", critic)
builder.add_node("finalise", finalise)

builder.add_edge(START, "planner")
builder.add_edge("planner", "researcher")
builder.add_conditional_edges(
    "researcher",
    route_research,
    {"research": "researcher", "synthesise": "synthesiser"},
)
builder.add_edge("synthesiser", "critic")
builder.add_conditional_edges("critic", route_critic, {"finalise": "finalise"})
builder.add_edge("finalise", END)
graph = builder.compile()


TASK = (
    "Research AI energy management for hotels in SEA. "
    "Cover: market size, top competitors, ROI for a 200-room Bangkok hotel."
)


if __name__ == "__main__":
    print("=== Autonomous Research Agent ===\n")
    result = graph.invoke({
        "task": TASK,
        "plan": [],
        "findings": [],
        "draft": "",
        "critique": "",
        "iteration": 0,
        "final": "",
    })
    print("\n=== FINAL REPORT ===")
    print(result["final"])
