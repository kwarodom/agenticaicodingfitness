#!/usr/bin/env python3
"""PART 9 · Real-world — a sovereign Smart-Hotel HVAC agent  [ADVANCED]

The payoff. A 500-room hotel runs 4 chillers, 40 AHUs, and 500 FCUs. A DGX Spark
on-site runs Gemma 4 with zero latency and zero cloud dependency — guest data and
building telemetry never leave the property (privacy by physical design, and
guaranteed compliance).

This is the production shape: one local agent clears the morning alarm queue.
It reads the alarms, pulls equipment status, and either resolves an energy issue
or dispatches maintenance — picking the right priority for a guest-impacting
fault vs. a routine one. Several tools, several rounds, all on-device.

Run:  python demos/step08_smart_hotel_mas.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import edgeview  # noqa: E402

# ── the building's live state (in-process; on a real site this is TimescaleDB) ─
_ALARMS = [
    {"id": "A1", "equipment": "chiller-2", "alarm": "condenser high pressure"},
    {"id": "A2", "equipment": "ahu-12",    "alarm": "filter differential pressure high"},
    {"id": "A3", "equipment": "fcu-412",   "alarm": "space temp high — guest room"},
]
_STATUS = {
    "chiller-2": {"condenser_pressure_psi": 298, "load_pct": 91, "guest_impact": "indirect"},
    "ahu-12":    {"filter_dp_inwc": 1.6, "supply_temp_f": 60, "guest_impact": "low"},
    "fcu-412":   {"valve_pct": 0, "room_temp_f": 79, "occupied": True, "guest_impact": "high"},
}
_ACTIONS: list[dict] = []

TOOLS = [
    {"type": "function", "function": {
        "name": "list_alarms",
        "description": "List all currently active HVAC alarms in the building. "
                       "Call this first to see the full queue.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_equipment_status",
        "description": "Get live status for one piece of equipment, including "
                       "whether it impacts guests.",
        "parameters": {"type": "object", "properties": {
            "equipment": {"type": "string"}}, "required": ["equipment"]}}},
    {"type": "function", "function": {
        "name": "dispatch_maintenance",
        "description": "Dispatch a technician. Use 'critical' for guest-impacting "
                       "faults, lower for routine maintenance.",
        "parameters": {"type": "object", "properties": {
            "equipment": {"type": "string"}, "issue": {"type": "string"},
            "priority": {"type": "string",
                         "enum": ["low", "medium", "high", "critical"]}},
            "required": ["equipment", "issue", "priority"]}}},
]


def list_alarms(args: dict) -> str:
    return json.dumps(_ALARMS)


def get_equipment_status(args: dict) -> str:
    eq = (args.get("equipment") or "").lower()
    return json.dumps(_STATUS.get(eq, {"error": f"unknown equipment {eq}"}))


def dispatch_maintenance(args: dict) -> str:
    _ACTIONS.append(args)
    return (f"Work order: {args.get('priority','?').upper()} — {args.get('equipment')}: "
            f"{args.get('issue')}. Technician dispatched.")


IMPLS = {"list_alarms": list_alarms, "get_equipment_status": get_equipment_status,
         "dispatch_maintenance": dispatch_maintenance}


def main() -> None:
    edgeview.banner("PART 9", "Real-world: sovereign Smart-Hotel HVAC agent", "ADVANCED")
    if not edgeview.require_local():
        return

    print("Scenario: 500-room hotel, DGX Spark on-site, Gemma 4 running locally.")
    print(f"Morning alarm queue: {len(_ALARMS)} active alarms across the building.")
    print("Guest telemetry never leaves the property — compliance by physical design.\n")
    print("Tools the local agent can call: list_alarms · get_equipment_status ·")
    print("dispatch_maintenance.  Watch it triage the whole queue on-device.\n")

    edgeview.sovereignty_line()
    messages = [
        {"role": "system", "content":
         "You are the on-premise HVAC operations agent for a 500-room hotel. "
         "Clear the alarm queue: list the alarms, check status for each, and "
         "dispatch maintenance with the correct priority. A guest-impacting fault "
         "is CRITICAL; a dirty filter is routine. Work through ALL alarms, then "
         "give a one-paragraph shift summary."},
        {"role": "user", "content":
         "Triage this morning's alarm queue and dispatch what's needed."},
    ]
    edgeview.call_with_tools(messages, TOOLS, IMPLS, max_rounds=8, max_tokens=1200)

    print(f"\n── Shift result: {len(_ACTIONS)} work order(s) dispatched ──")
    for a in _ACTIONS:
        print(f"  • [{a.get('priority','?').upper():<8}] {a.get('equipment'):<10} "
              f"{a.get('issue')}")

    print("\nTakeaway: this is the work that used to need a control-room operator —")
    print("now one sovereign agent clears it in a single run, with the right priority")
    print("for the guest-impacting room, and not a byte sent to the cloud.")


if __name__ == "__main__":
    main()
