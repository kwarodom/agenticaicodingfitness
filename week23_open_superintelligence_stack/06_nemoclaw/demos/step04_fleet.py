#!/usr/bin/env python3
"""PART 4 · A fleet of specialists  [ADVANCED]

One NemoClaw specialist is useful; a FLEET is a system. A supervisor routes each task
to the right authored expert — HVAC, finance, or code — each its own persona + skills
+ signed policy, each in its own OpenShell sandbox. This is the multi-agent pattern
(Week 9/16), now built from real specialists. This demo routes a task in SIM as a
trace, and in REAL asks the base model to pick the specialist (router pattern).

Run:  python demos/step04_fleet.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import view  # noqa: E402

# The fleet — each a NemoClaw specialist (base model + persona + skills + policy).
FLEET = {
    "hvac":    ("nemotron-3-nano", "comfort/energy from telemetry", ["get_runbook", "optimize_setpoints"]),
    "finance": ("nemotron-3-super", "budgets, invoices, forecasts", ["query_ledger", "run_forecast"]),
    "code":    ("nemotron-3-super", "reads/writes/reviews code", ["read_repo", "run_tests"]),
}
TASK = "Room 1203 keeps overheating and guests are complaining — investigate and act."


def _route(task: str) -> str:
    """A tiny keyword router (the supervisor's job) — REAL mode uses the model instead."""
    t = task.lower()
    if any(k in t for k in ("room", "hvac", "temp", "cooling", "overheat")):
        return "hvac"
    if any(k in t for k in ("budget", "invoice", "forecast", "cost")):
        return "finance"
    return "code"


def main() -> None:
    view.banner("PART 4", "A fleet of specialists", "ADVANCED")
    view.mode_line()

    print("A supervisor orchestrates a fleet of NemoClaw specialists (each authored + sandboxed):")
    for name, (model, role, tools) in FLEET.items():
        print(f"  • {name:8s} [{model:16s}] {role:32s} tools: {', '.join(tools)}")
    print(f"\nIncoming task: {TASK!r}\n")

    if view.is_sim():
        print("SIM — the supervisor → specialist routing (REAL mode asks the model to route):")
        print("  → supervisor inspects the task")
        print("  ~ reason: mentions 'room' + 'overheating' → an HVAC concern")
        print("  → routes to the HVAC specialist (nemotron-3-nano)")
        print("  · HVAC specialist runs in its OpenShell sandbox, dispatches maintenance")
        print("  ← supervisor collects the result and reports to the user")
        pick = _route(TASK)
    else:
        names = ", ".join(FLEET)
        got = view.classify(
            f"You are a supervisor over specialist agents [{names}]. Answer with the single "
            f"specialist name best suited to this task.\nTask: {TASK}",
            labels=list(FLEET), title="supervisor routes the task")
        pick = got or _route(TASK)
        print(f"\n  → supervisor routed to: {pick} specialist"
              + ("" if got else "   (model gave no name — heuristic fallback)"))

    model, role, tools = FLEET[pick]
    print(f"\n  Selected specialist: {pick}  [{model}] — {role}")
    print(f"  It handles the task with its skills/tools ({', '.join(tools)}) in its own sandbox.")

    print("\nTakeaway: NemoClaw builds the specialists; a supervisor composes them into a fleet.")
    print("That's the Week 9/16 multi-agent system — a team of authored domain experts.")


if __name__ == "__main__":
    main()
