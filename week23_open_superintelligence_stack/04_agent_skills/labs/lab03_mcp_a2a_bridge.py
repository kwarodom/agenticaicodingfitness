#!/usr/bin/env python3
"""LAB 03 · One skill, two transports — the SAME tool over MCP and via an A2A card.

Lab 02 wired the hotel-telemetry skill straight into one agent. Real fleets need
standards: MCP moves TOOLS (agent→tools) and A2A moves TASKS (agent→agent). Here
you drive both by hand: real JSON-RPC frames dispatched to a real in-process tool,
then an Agent Card + the six-state A2A task lifecycle for the same capability.
Everything executes locally and offline — the frames are genuine, only the wire
(stdio/HTTP) is skipped.

Run:  .venv/bin/python week23/04_agent_skills/labs/lab03_mcp_a2a_bridge.py
Needs: nothing — stdlib only, no endpoint, terminates in seconds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402  (imported for house consistency; lab is offline)

SENSORS = {"1203": {"temp_c": 29.4, "setpoint_c": 23.0, "alarm": "HIGH_TEMP"}}
TOOL = {"name": "query_room_telemetry",
        "description": "Skill: hotel-telemetry — read live sensors for one room.",
        "inputSchema": {"type": "object",
                        "properties": {"room": {"type": "string"}}, "required": ["room"]}}


def call_tool(name: str, args: dict) -> dict:
    if name == "query_room_telemetry":
        return SENSORS.get(str(args.get("room", "")), {"error": "unknown room"})
    return {"error": f"no such tool {name}"}


# ── an in-process MCP server: same dispatch a stdio server does, minus the pipe ──
def mcp_handle(req: dict) -> dict:
    m = req.get("method")
    if m == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "hotel-telemetry-skill", "version": "1.0.0"}}
    elif m == "tools/list":
        result = {"tools": [TOOL]}
    elif m == "tools/call":
        p = req.get("params", {})
        out = call_tool(p.get("name", ""), p.get("arguments", {}))
        result = {"content": [{"type": "text", "text": json.dumps(out)}]}
    else:
        return {"jsonrpc": "2.0", "id": req.get("id"),
                "error": {"code": -32601, "message": f"unknown method {m}"}}
    return {"jsonrpc": "2.0", "id": req.get("id"), "result": result}


def rpc(req: dict) -> None:
    print("  → " + json.dumps(req))
    print("  ← " + json.dumps(mcp_handle(req)) + "\n")


AGENT_CARD = {  # served at /.well-known/agent.json by a real A2A agent
    "name": "hotel-telemetry-agent",
    "description": "Specialist agent wrapping the hotel-telemetry skill: reads room "
                   "sensors, flags out-of-band readings per SOP-HVAC-07.",
    "url": "http://spark.local:9999/a2a",
    "version": "1.0.0",
    "capabilities": {"streaming": True, "pushNotifications": False},
    "skills": [{"id": "hotel-telemetry", "name": "Hotel telemetry",
                "description": "Query room sensors; recommend work orders.",
                "examples": ["Room 1203 feels hot — what's going on?"]}],
}


def main() -> None:
    print("━" * 64)
    print("  LAB 03 — One skill over MCP (tools) and A2A (agents)   [OFFLINE]")
    print("━" * 64 + "\n")

    print("▣ PART A · MCP — agent→TOOLS. Drive the handshake by hand (JSON-RPC 2.0):\n")
    rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "clientInfo": {"name": "lab03"}}})
    rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "query_room_telemetry", "arguments": {"room": "1203"}}})
    print("  ✓ that tools/call really executed — 29.4°C came from the tool, not a canned")
    print("    string. Any MCP client (Claude Code, NAT, your own loop) speaks these")
    print("    exact frames over stdio or HTTP.\n")

    print("▣ PART B · A2A — agent→AGENT. First, discovery: the Agent Card a peer fetches")
    print("  from /.well-known/agent.json:\n")
    for line in json.dumps(AGENT_CARD, indent=2).splitlines():
        print("  │ " + line)
    print()

    print("▣ PART B2 · the A2A task lifecycle — a peer delegates, states advance:\n")
    task = {"id": "task-7f3a", "state": "submitted",
            "message": "Investigate the high-temp alarm in room 1203."}
    print(f"  ◈ state: {task['state']:<14} — peer POSTs the task to the agent")
    task["state"] = "working"
    print(f"  ◈ state: {task['state']:<14} — agent loads its hotel-telemetry skill…")
    reading = call_tool("query_room_telemetry", {"room": "1203"})
    print(f"      (internally it may itself use MCP: reading = {json.dumps(reading)})")
    task["state"] = "completed"
    task["artifact"] = (f"Room 1203: {reading['temp_c']}°C vs {reading['setpoint_c']}°C "
                        f"setpoint, alarm {reading['alarm']} — recommend work order per SOP-HVAC-07.")
    print(f"  ◈ state: {task['state']:<14} — artifact returned to the delegating agent:")
    print(f"      {task['artifact']}")
    print("  · full state machine: submitted → working → input-required → completed /")
    print("    failed / canceled  (Week 17 covers all six)\n")

    print("The split, one line each:")
    print("  → MCP:  frontier agent ──tools──▶ this skill's query_room_telemetry")
    print("  → A2A:  frontier agent ──task───▶ a specialist agent that LOADED the skill\n")
    print("Takeaway: the skill file never changed between labs 01→03 — one capability,")
    print("carried by two open standards. Compare demos/step04_skills_mcp_a2a.py, then")
    print("go to 05_aiq_research_lab, where AI-Q wires skills as tools at scale.")


if __name__ == "__main__":
    main()
