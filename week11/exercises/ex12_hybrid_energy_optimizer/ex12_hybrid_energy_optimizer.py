"""
Exercise 12 — Hybrid Energy Optimizer (Advanced).

NOTE: PDF specifies CrewAI + LangGraph; this implementation uses raw
Anthropic SDK (in place of CrewAI) + LangGraph, per project preference.
The 'analysis crew' becomes a sequential dataclass-Agent pipeline.

Phase 1: Two agents (Analyst -> OptimisationEngineer) analyse zone-level
sensor data and emit optimisation commands.
Phase 2: A LangGraph execution graph validates those commands, applies
them, monitors impact, and can rollback if savings underperform.

Key concepts:
    - Framework composition — analysis output feeds LangGraph state
    - Execution graph with rollback arc
    - Simulated actuator calls (pattern for real REST API integration)
    - Multi-phase pipeline: analyse -> validate -> execute -> monitor
"""

import json
import os
from dataclasses import dataclass
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
import anthropic

load_dotenv()
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
MODEL = "claude-haiku-4-5-20251001"
client = anthropic.Anthropic()


SENSOR_DATA = {
    "hotel": "Grand Suites Bangkok",
    "zones": {
        "3F_rooms": {"kw": 48, "occ": 0.60, "setpoint": 24},
        "4F_rooms": {"kw": 45, "occ": 0.85, "setpoint": 24},
        "lobby": {"kw": 35, "occ": 0.90, "setpoint": 22},
        "pool": {"kw": 28, "occ": 0.20, "setpoint": 30},
        "conf_hall": {"kw": 22, "occ": 0.00, "setpoint": 23},
    },
    "total_kw": 178,
    "target_kw": 155,
}


# --- Phase 1: Sequential Agent Analysis ---


@dataclass
class Agent:
    role: str
    goal: str
    backstory: str

    def system(self) -> str:
        return (
            f"You are a {self.role}. {self.backstory}\n"
            f"Your goal: {self.goal}"
        )

    def run(self, task: str, max_tokens: int = 800) -> str:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=self.system(),
            messages=[{"role": "user", "content": task}],
        )
        return resp.content[0].text


def run_analysis(data: dict) -> str:
    analyst = Agent(
        "Energy Analyst",
        "Find the top 3 HVAC zones to optimise",
        "Expert in hotel BMS and occupancy-based scheduling.",
    )
    optimiser = Agent(
        "Optimisation Engineer",
        "Generate safe HVAC setpoint adjustments",
        "Controls engineer — never sacrifices guest comfort.",
    )

    a_out = analyst.run(
        f"Analyse this hotel sensor snapshot:\n{json.dumps(data)}\n"
        "Identify 3 zones with highest savings potential. "
        "List zone, current kW, occupancy, savings estimate."
    )
    o_out = optimiser.run(
        "Generate optimisation commands for the 3 zones. "
        "Format each command as a single line: "
        "ZONE|ACTION|OLD_VAL|NEW_VAL|EST_SAVINGS_PCT\n\n"
        f"=== ANALYSIS ===\n{a_out}\n=== END ==="
    )
    return o_out


# --- Phase 2: LangGraph Execution ---


class ExecState(TypedDict):
    commands: str
    executed: list
    savings_pct: float
    rollback: bool
    status: str


def validate_and_execute(state: ExecState) -> ExecState:
    lines = [
        l.strip() for l in state["commands"].splitlines() if "|" in l
    ]
    done: list = []
    for line in lines[:3]:
        parts = line.split("|")
        zone = parts[0]
        action = parts[1] if len(parts) > 1 else "adjust"
        print(f"  [Execute] {zone}: {action}")
        done.append({"zone": zone, "action": action, "ok": True})
    savings = min(len(done) * 7.0, 22.0)
    return {
        "executed": done,
        "savings_pct": savings,
        "rollback": savings < 4,
    }


def monitor(state: ExecState) -> ExecState:
    s = state["savings_pct"]
    print(f"[Monitor] Energy reduced {s:.1f}%")
    return {"status": "success" if s >= 10 else "low_savings"}


def rollback_fn(state: ExecState) -> ExecState:
    print("[Rollback] Restoring all setpoints to default.")
    return {"executed": [], "savings_pct": 0.0, "status": "rolled_back"}


def route_exec(state: ExecState) -> str:
    return "rollback" if state["rollback"] else "monitor"


builder = StateGraph(ExecState)
builder.add_node("execute", validate_and_execute)
builder.add_node("monitor", monitor)
builder.add_node("rollback", rollback_fn)
builder.add_edge(START, "execute")
builder.add_conditional_edges(
    "execute",
    route_exec,
    {"monitor": "monitor", "rollback": "rollback"},
)
builder.add_edge("monitor", END)
builder.add_edge("rollback", END)
graph = builder.compile()


if __name__ == "__main__":
    print("Phase 1 — analysis pipeline...")
    cmds = run_analysis(SENSOR_DATA)
    print(f"Commands generated:\n{cmds}\n")

    print("Phase 2 — LangGraph execution...")
    result = graph.invoke({
        "commands": cmds,
        "executed": [],
        "savings_pct": 0.0,
        "rollback": False,
        "status": "starting",
    })
    print(f"\nFinal status : {result['status']}")
    print(f"Energy saved : {result['savings_pct']:.1f}%")
    print(f"Actions done : {len(result['executed'])}")
