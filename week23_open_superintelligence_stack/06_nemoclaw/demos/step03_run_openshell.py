#!/usr/bin/env python3
"""PART 3 · Run it safely in OpenShell  [INTERMEDIATE]

An equipped specialist is powerful, so NemoClaw runs it inside an OpenShell sandbox
(App 7) under a SIGNED policy with an egress allowlist. Every tool call is checked by
a policy gateway BEFORE it executes in the sandbox. This demo runs a task and shows
the gate: agent → tool (call) → policy check (ok/deny) → sandbox execute → result.

Run:  python demos/step03_run_openshell.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import view  # noqa: E402

# The signed policy the specialist runs under (enforced by OpenShell at runtime).
POLICY = {
    "signed_by": "ops-team",
    "allow_tools": ["get_room_telemetry", "dispatch_work_order"],
    "egress_allowlist": ["telemetry.internal"],
    "deny": ["exfiltrate_data", "public_http"],
}

TOOLS = [
    {"type": "function", "function": {
        "name": "get_room_telemetry", "description": "Read live HVAC telemetry (allowed).",
        "parameters": {"type": "object", "properties": {"room": {"type": "string"}},
                       "required": ["room"]}}},
    {"type": "function", "function": {
        "name": "dispatch_work_order", "description": "Dispatch maintenance at a priority (allowed).",
        "parameters": {"type": "object", "properties": {
            "room": {"type": "string"}, "priority": {"type": "string", "enum": ["CRITICAL", "ROUTINE"]}},
            "required": ["room", "priority"]}}},
]
ROOMS = {"1203": {"temp_c": 26.4, "setpoint_c": 22.0, "occupied": True}}


def _policy_check(name: str) -> bool:
    """OpenShell policy gateway — a tool runs only if the signed policy allows it."""
    return name in POLICY["allow_tools"]


def _impl(name, args):
    room = str(args.get("room") or "").strip()
    if name == "get_room_telemetry":
        return json.dumps(ROOMS.get(room, {}))
    return json.dumps({"work_order": "WO-" + room, "priority": args.get("priority"),
                       "status": "dispatched"})


def main() -> None:
    view.banner("PART 3", "Run it safely in OpenShell", "INTERMEDIATE")
    view.mode_line()

    print("The specialist runs inside an OpenShell sandbox under this SIGNED policy:")
    print(json.dumps(POLICY, indent=2))
    print("\nEvery tool call is checked by the policy gateway BEFORE it executes:\n")

    if view.is_sim():
        print("SIM — a guarded task run (REAL mode executes the loop):")
        print("  → agent calls get_room_telemetry({'room':'1203'})")
        print("  ✓ POLICY get_room_telemetry allowed → execute in sandbox")
        print("  ← {'temp_c':26.4,'setpoint_c':22.0,'occupied':true}")
        print("  ~ reason: occupied + >3°C from setpoint → guest-impacting → CRITICAL")
        print("  → agent calls dispatch_work_order({'room':'1203','priority':'CRITICAL'})")
        print("  ✓ POLICY dispatch_work_order allowed → execute in sandbox")
        print("  ← {'work_order':'WO-1203','status':'dispatched'}")
        print("  ⚠ (a call to public_http would be DENIED — egress allowlist blocks it)")
        print("  · answer: Dispatched CRITICAL maintenance for room 1203 — all within policy.")
    else:
        client = view._client()
        messages = [{"role": "system", "content": "You are an HVAC specialist in a sandbox. Use the "
                     "tools; an occupied room >3°C from setpoint = CRITICAL. Pass the room as its "
                     "bare id (e.g. '1203'). If CRITICAL you MUST dispatch_work_order before answering."},
                    {"role": "user", "content": "Room 1203 alarm — triage and act."}]
        for _ in range(4):
            try:
                r = client.chat.completions.create(model=config.MODEL, messages=messages,
                                                   tools=TOOLS, max_tokens=400, temperature=0.2)
            except Exception as e:  # noqa: BLE001
                view._endpoint_error(e); return
            msg = r.choices[0].message
            calls = msg.tool_calls or []
            if not calls:
                print("  · answer:", (msg.content or "").strip()[:300]); break
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name,
                                                          "arguments": c.function.arguments}} for c in calls]})
            for c in calls:
                args = json.loads(c.function.arguments or "{}")
                if not _policy_check(c.function.name):
                    res = json.dumps({"error": "POLICY DENY — tool not in signed allowlist"})
                    print(f"  ✕ POLICY {c.function.name} DENIED")
                else:
                    res = _impl(c.function.name, args)
                    print(f"  ✓ POLICY {c.function.name} allowed → execute in sandbox")
                    print(f"  → ACT {c.function.name}({args})")
                    print(f"  ← OBSERVE {res}")
                messages.append({"role": "tool", "tool_call_id": c.id, "content": res})

    print("\nTakeaway: capability without containment is a liability. OpenShell's signed")
    print("policy + egress allowlist let a powerful specialist act — but only within bounds.")


if __name__ == "__main__":
    main()
