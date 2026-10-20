#!/usr/bin/env python3
"""PART 4 · Tool-calling — Nemotron as a sub-agent  [INTERMEDIATE]

"Agent = Model + Harness." Nemotron 3 has native tool-calling: give it tool schemas
and it emits structured calls. That's how it becomes a sub-agent in a multi-agent
system (the AI-Q blueprint uses Nano for routing/sub-agents, Super to orchestrate).
This demo wires two real Python tools and, in REAL mode, lets the model call them.

Run:  python demos/step04_tool_calling.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import ntview  # noqa: E402

TOOLS = [
    {"type": "function", "function": {
        "name": "get_room_telemetry", "description": "Read live HVAC telemetry for a room.",
        "parameters": {"type": "object", "properties": {"room": {"type": "string"}}, "required": ["room"]}}},
    {"type": "function", "function": {
        "name": "dispatch_work_order", "description": "Dispatch maintenance at a priority.",
        "parameters": {"type": "object", "properties": {
            "room": {"type": "string"}, "priority": {"type": "string", "enum": ["CRITICAL", "ROUTINE"]}},
            "required": ["room", "priority"]}}},
]
ROOMS = {"1203": {"temp_c": 26.4, "setpoint_c": 22.0, "occupied": True}}


def _room_key(room) -> str:
    """Normalize room args the model might phrase as 'Room 1203', ' 1203 ', 'rm 1203'."""
    import re
    return re.sub(r"(?i)^\s*(?:room|rm)?\s*#?\s*", "", str(room or "")).strip()


def _impl(name, args):
    room = _room_key(args.get("room"))
    if name == "get_room_telemetry":
        return json.dumps(ROOMS.get(room, {}))
    return json.dumps({"work_order": "WO-" + room, "priority": args.get("priority"), "status": "dispatched"})


def main() -> None:
    ntview.banner("PART 4", "Tool-calling — Nemotron as a sub-agent", "INTERMEDIATE")
    ntview.mode_line()

    print("Two real Python tools wired to the model (a sovereign HVAC sub-agent):")
    print("  • get_room_telemetry(room)   • dispatch_work_order(room, priority)\n")

    if ntview.is_sim():
        print("SIM — the tool-calling loop the model runs (REAL mode executes it):")
        print("  → model calls get_room_telemetry({'room':'1203'})")
        print("  ← {'temp_c':26.4,'setpoint_c':22.0,'occupied':true}")
        print("  ~ reason: occupied + >3°C from setpoint → guest-impacting → CRITICAL")
        print("  → model calls dispatch_work_order({'room':'1203','priority':'CRITICAL'})")
        print("  ← {'work_order':'WO-1203','status':'dispatched'}")
        print("  · answer: Dispatched CRITICAL maintenance for room 1203.")
    else:
        client = ntview._client()
        messages = [{"role": "system", "content": "You are an HVAC sub-agent. Use the tools; "
                     "occupied room >3°C from setpoint = CRITICAL. Pass the room as its bare id "
                     "(e.g. '1203', not 'Room 1203'). After reading telemetry, if it is CRITICAL "
                     "you MUST call dispatch_work_order before answering."},
                    {"role": "user", "content": "Room 1203 alarm — triage and act."}]
        for _ in range(4):
            try:
                r = client.chat.completions.create(model=config.MODEL, messages=messages,
                                                   tools=TOOLS, max_tokens=400, temperature=0.2)
            except Exception as e:  # noqa: BLE001
                ntview._endpoint_error(e); return
            msg = r.choices[0].message
            calls = msg.tool_calls or []
            if not calls:
                print("  · answer:", (msg.content or "").strip()[:300]); break
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
            for c in calls:
                args = json.loads(c.function.arguments or "{}")
                res = _impl(c.function.name, args)
                print(f"  → ACT {c.function.name}({args})")
                print(f"  ← OBSERVE {res}")
                messages.append({"role": "tool", "tool_call_id": c.id, "content": res})

    print("\nTakeaway: native tool-calling turns a model into an agent. Nano handles")
    print("many cheap sub-agents; Super orchestrates them — the multi-agent pattern.")


if __name__ == "__main__":
    main()
