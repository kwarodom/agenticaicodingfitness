#!/usr/bin/env python3
"""PART 8 · A sovereign agent — local function calling  [INTERMEDIATE]

An agent is a model that can ACT on your systems, not just talk. Gemma 4 has
native function calling: give it tool schemas and it emits structured JSON tool
calls. The whole agent loop — REASON → ACT (call your tool) → OBSERVE (result) →
repeat — runs on YOUR hardware, against YOUR systems, with zero cloud round-trips.

This demo wires two real Python functions (read a sensor, dispatch maintenance)
to the local model and asks it to diagnose a chiller fault and take action. Watch
it decide which tool to call, see the result feed back, and finish with a plan.

Run:  python demos/step07_tool_calling_agent.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import edgeview  # noqa: E402

# ── a tiny "building" the tools read/write (all in-process, all local) ────────
_SENSORS = {
    "chiller-2": {"condenser_pressure_psi": 295, "approach_temp_f": 18.5,
                  "status": "alarm: condenser high pressure", "setpoint_f": 44},
    "ahu-3":     {"filter_dp_inwc": 1.4, "supply_temp_f": 61, "status": "ok"},
}
_DISPATCHED: list[dict] = []

TOOLS = [
    {"type": "function", "function": {
        "name": "get_sensor_reading",
        "description": "Read the live sensor data and status for one piece of HVAC "
                       "equipment. Use this first to diagnose before acting.",
        "parameters": {"type": "object", "properties": {
            "equipment_id": {"type": "string",
                             "description": "e.g. 'chiller-2' or 'ahu-3'"}},
            "required": ["equipment_id"]}}},
    {"type": "function", "function": {
        "name": "dispatch_maintenance",
        "description": "Dispatch a maintenance technician to equipment with a "
                       "priority. Only call this once you've diagnosed the issue.",
        "parameters": {"type": "object", "properties": {
            "equipment_id": {"type": "string"},
            "issue": {"type": "string", "description": "short diagnosis"},
            "priority": {"type": "string",
                         "enum": ["low", "medium", "high", "critical"]}},
            "required": ["equipment_id", "issue", "priority"]}}},
]


def get_sensor_reading(args: dict) -> str:
    eq = (args.get("equipment_id") or "").lower()
    data = _SENSORS.get(eq)
    return json.dumps(data) if data else f"No equipment '{eq}'. Known: {list(_SENSORS)}"


def dispatch_maintenance(args: dict) -> str:
    _DISPATCHED.append(args)
    return (f"Work order created: {args.get('priority','?').upper()} priority for "
            f"{args.get('equipment_id')} — '{args.get('issue')}'. Tech notified.")


IMPLS = {"get_sensor_reading": get_sensor_reading,
         "dispatch_maintenance": dispatch_maintenance}


def main() -> None:
    edgeview.banner("PART 8", "A sovereign agent (local function calling)", "INTERMEDIATE")
    if not edgeview.require_local():
        return

    print("Registered two LOCAL tools the model can call:")
    print("  • get_sensor_reading(equipment_id)")
    print("  • dispatch_maintenance(equipment_id, issue, priority)")
    print("The model has never seen the sensor data — it must ACT to get it.\n")

    edgeview.sovereignty_line()
    messages = [
        {"role": "system", "content":
         "You are a sovereign HVAC operations agent running on-premise. Diagnose "
         "issues using the sensor tool, then dispatch maintenance with an "
         "appropriate priority. Be decisive and concise."},
        {"role": "user", "content":
         "Chiller-2 just raised an alarm. Investigate and take the right action."},
    ]
    out = edgeview.call_with_tools(messages, TOOLS, IMPLS, max_rounds=4)

    print(f"\nWork orders created this run: {len(_DISPATCHED)}")
    for wo in _DISPATCHED:
        print(f"  • [{wo.get('priority','?').upper()}] {wo.get('equipment_id')}: "
              f"{wo.get('issue')}")

    print("\nTakeaway: the model didn't just describe the fault — it READ a live")
    print("sensor and DISPATCHED a work order, all through tools running on your")
    print("hardware. That's a sovereign agent: it acts, and nothing leaves the edge.")


if __name__ == "__main__":
    main()
