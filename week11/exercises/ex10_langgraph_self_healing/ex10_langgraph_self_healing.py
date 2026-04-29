"""
Exercise 10 — LangGraph Self-Healing Data Pipeline (Advanced).

Sensor readings are ingested, validated against domain rules, and — when
invalid — passed to an AI healer node. The healer uses Claude to infer
corrected values and routes back for re-validation, up to 3 retries.
Failure triggers an alert.

Key concepts:
    - Cycle with bounded retries (retry_count guard)
    - AI-powered data repair via Claude with JSON output
    - Multi-route conditional edge (store / heal / fail)
    - Injecting domain knowledge through the prompt
"""

import json
import os
import random
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic

load_dotenv()
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")


class Pipeline(TypedDict):
    raw: dict
    clean: Optional[dict]
    errors: list
    retries: int
    log: list
    status: str


def ingest(state: Pipeline) -> Pipeline:
    data = {
        "hotel": "ALTO_BKK_001",
        "floor": random.choice([3, 4, "unknown", None]),
        "hvac_temp_c": random.choice([22.5, -99.9, None, 999]),
        "power_kw": random.choice([45.2, -5.0, "ERROR", None]),
        "occupancy": random.choice([75, 150, -10, None]),
    }
    print(f"[Ingest] {data}")
    return {"raw": data, "status": "validating"}


def validate(state: Pipeline) -> Pipeline:
    d = state["raw"]
    errs: list = []
    if d.get("floor") not in range(1, 21):
        errs.append(f"floor invalid: {d.get('floor')}")
    t = d.get("hvac_temp_c")
    if not isinstance(t, (int, float)) or not (10 <= t <= 40):
        errs.append(f"temp out of range: {t}")
    p = d.get("power_kw")
    if not isinstance(p, (int, float)) or p < 0 or p > 500:
        errs.append(f"power invalid: {p}")
    o = d.get("occupancy")
    if not isinstance(o, (int, float)) or not (0 <= o <= 100):
        errs.append(f"occupancy invalid: {o}")
    print(f"[Validate] {len(errs)} errors")
    return {
        "errors": errs,
        "clean": state["raw"] if not errs else None,
    }


def ai_heal(state: Pipeline) -> Pipeline:
    prompt = (
        "You are an IoT data healer. Fix only invalid sensor fields.\n"
        f"Raw data: {json.dumps(state['raw'])}\n"
        f"Errors: {state['errors']}\n"
        "Rules: floor 1-20, hvac_temp_c 10-40, power_kw 0-500, "
        "occupancy 0-100. Use null for unrecoverable fields.\n"
        "Return ONLY valid JSON, no markdown."
    )
    result = llm.invoke(prompt)
    try:
        text = result.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        fixed = json.loads(text)
        print(f"[Heal] Fixed: {fixed}")
        return {
            "raw": fixed,
            "log": state["log"]
            + [f"Healed attempt {state['retries'] + 1}"],
            "retries": state["retries"] + 1,
            "status": "validating",
        }
    except (json.JSONDecodeError, IndexError) as e:
        print(f"[Heal] Parse error: {e}")
        return {
            "retries": state["retries"] + 1,
            "status": "validating",
        }


def store(state: Pipeline) -> Pipeline:
    print(f"[Store] Saved: {state['clean']}")
    return {"status": "complete"}


def alert(state: Pipeline) -> Pipeline:
    print(
        f"[Alert] Pipeline FAILED after {state['retries']} attempts — "
        "paging data team."
    )
    return {"status": "failed"}


def route(state: Pipeline) -> str:
    if not state["errors"]:
        return "store"
    if state["retries"] >= 3:
        return "fail"
    return "heal"


builder = StateGraph(Pipeline)
builder.add_node("ingest", ingest)
builder.add_node("validate", validate)
builder.add_node("heal", ai_heal)
builder.add_node("store", store)
builder.add_node("alert", alert)
builder.add_edge(START, "ingest")
builder.add_edge("ingest", "validate")
builder.add_conditional_edges(
    "validate",
    route,
    {"store": "store", "heal": "heal", "fail": "alert"},
)
builder.add_edge("heal", "validate")
builder.add_edge("store", END)
builder.add_edge("alert", END)
graph = builder.compile()


if __name__ == "__main__":
    for i in range(3):
        print(f"\n--- Run {i + 1} ---")
        res = graph.invoke({
            "raw": {},
            "clean": None,
            "errors": [],
            "retries": 0,
            "log": [],
            "status": "ingesting",
        })
        print(f"Status: {res['status']}")
