#!/usr/bin/env python3
"""LAB 02 · Load YOUR skill into a live frontier agent — the tool-call loop, timed.

Lab 01 authored `hotel-telemetry` as files. Here a real model on YOUR endpoint
(Spark/Ollama/build.nvidia.com) discovers it, invokes its tools, answers — timed.

Run:  .venv/bin/python week23/04_agent_skills/labs/lab02_load_skill_live.py
Endpoint via config.py (DGX_CONN=local|tunnel|cloud + DGX_BASE_URL / DGX_API_KEY);
no endpoint → real setup commands plus a labeled expected-output sample.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SKILL_MD = (config.SANDBOX / "skills" / "hotel-telemetry" / "SKILL.md")

TOOLS = [{"type": "function", "function": {
    "name": "query_room_telemetry",
    "description": "Skill: hotel-telemetry — read live sensors for one hotel room.",
    "parameters": {"type": "object",
                   "properties": {"room": {"type": "string"}}, "required": ["room"]}}},
    {"type": "function", "function": {
        "name": "file_work_order",
        "description": "Skill: hotel-telemetry — file a maintenance work order for a room.",
        "parameters": {"type": "object", "properties": {
            "room": {"type": "string"}, "summary": {"type": "string"}},
            "required": ["room", "summary"]}}}]

SENSORS = {"1203": {"temp_c": 29.4, "setpoint_c": 23.0, "alarm": "HIGH_TEMP"}}


def tool_impl(name: str, args: dict) -> str:
    if name == "query_room_telemetry":
        return json.dumps(SENSORS.get(str(args.get("room", "")), {"error": "unknown room"}))
    if name == "file_work_order":
        return json.dumps({"work_order": "WO-4211", "status": "filed", "sop": "SOP-HVAC-07"})
    return json.dumps({"error": f"no such tool {name}"})


def expected_output() -> None:
    print("[no endpoint — showing expected output] a REAL run looks like:\n")
    print("  → ACT query_room_telemetry({'room': '1203'})")
    print('  ← OBSERVE {"temp_c": 29.4, "setpoint_c": 23.0, "alarm": "HIGH_TEMP"}')
    print("  → ACT file_work_order({'room': '1203', 'summary': '29.4C vs 23C, HIGH_TEMP'})")
    print('  ← OBSERVE {"work_order": "WO-4211", "status": "filed", "sop": "SOP-HVAC-07"}')
    print("  · ANSWER: Room 1203 reads 29.4°C vs 23°C, HIGH_TEMP — WO-4211 filed per SOP-HVAC-07.")
    print("  ◆ 2 tool hop(s) in ~5–45 s (model + hardware dependent)\n")
    print("Bring an endpoint up, then rerun this lab:")
    print("  ollama pull qwen3.6:35b-a3b-q8_0       # C 💻 local: any tool-capable model")
    print("  # A 🖥️ your Spark over Tailscale")
    print("  export DGX_CONN=tunnel DGX_TUNNEL_URL=http://<spark>.<tailnet>.ts.net:11434/v1")
    print("  # B ☁️ build.nvidia.com (usage-billed, not sovereign)")
    print("  export DGX_CONN=cloud DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1 DGX_API_KEY=nvapi-...")


def main() -> None:
    print("━" * 64)
    print("  LAB 02 — Load a skill into a LIVE frontier agent   [ENDPOINT]")
    print("━" * 64 + "\n")
    print(f"▣ connection: {config.CONN} ({config.conn_human()}) · endpoint: {config.safe_base_url()}")
    print(f"  mode: {config.MODE.upper()} · model: {config.MODEL} · {config.cost_note()}\n")

    # ▣ STEP 1 — discover: read the skill's metadata (from lab01's files if present)
    print("▣ STEP 1 · DISCOVER — the agent reads the skill file first:")
    if SKILL_MD.exists():
        body = SKILL_MD.read_text()
        print(f"  ✓ found {SKILL_MD.relative_to(config.PKG)} (authored in lab01)")
    else:
        body = ("---\nname: hotel-telemetry\ndescription: Query room sensors; use for room"
                " temperature/alarm questions.\n---\nAlways call the tool; cite the reading.")
        print("  ◈ lab01 not run yet — using an inline copy of the same skill.")
    for line in body.splitlines()[:8]:
        print("  │ " + line)
    print("  │ …\n")

    if config.MODE != "real":
        expected_output()
        return

    # ▣ STEP 2 — invoke: the model runs the loop for real, on YOUR endpoint
    print("▣ STEP 2 · INVOKE — live tool-call loop (max 4 hops, 60 s/request, 90 s total):")
    from openai import OpenAI
    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=60.0, max_retries=0)
    messages = [{"role": "system", "content": "You have loaded this skill:\n" + body +
                 "\nFollow it exactly: call the tool before answering; be concise."},
                {"role": "user", "content": "Room 1203 feels hot — what's going on?"}]
    t0, hops = time.time(), 0
    for _ in range(4):
        if time.time() - t0 > 90:
            print("  ✗ 90 s loop budget spent — endpoint is up but slow; try a faster model.")
            break
        try:
            r = client.chat.completions.create(model=config.MODEL, messages=messages,
                                               tools=TOOLS, max_tokens=300, temperature=0.2)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ request failed ({type(e).__name__}: {e}) — check the endpoint,")
            print("    the /v1 suffix, and that the model supports tool calls.")
            return
        msg = r.choices[0].message
        calls = msg.tool_calls or []
        if not calls:
            print("  · ANSWER:", (msg.content or "").strip()[:400]
                  or "(model returned empty content — try a larger tool-capable model)")
            break
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name,
                                                      "arguments": c.function.arguments}}
                                        for c in calls]})
        for c in calls:
            hops += 1
            args = json.loads(c.function.arguments or "{}")
            res = tool_impl(c.function.name, args)
            print(f"  → ACT {c.function.name}({args})")
            print(f"  ← OBSERVE {res}")
            messages.append({"role": "tool", "tool_call_id": c.id, "content": res})
    print(f"\n  ◆ {hops} tool hop(s) in {time.time() - t0:.1f}s · {config.cost_note()}")
    print("\nTakeaway: the agent gained YOUR capability from a file — its code never")
    print("changed. Next lab: the same skill rides MCP and A2A (lab03).")


if __name__ == "__main__":
    main()
