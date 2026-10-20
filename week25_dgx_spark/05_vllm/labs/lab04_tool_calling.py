#!/usr/bin/env python3
"""Lab 05-4 · Tool calling through vLLM's OpenAI API: one full round trip with an agent-ready model.

The playbook's agent-ready recipe starts nvidia/Qwen3.6-35B-A3B-NVFP4 with
`--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3`. Those flags make vLLM
turn the model's own tool-call syntax into the OpenAI `tool_calls` field, so any OpenAI client can drive an
agent loop. This lab runs that loop once:

  user question → model returns tool_calls → THIS script runs the tool (a fake hotel sensor, clearly local)
  → tool result goes back as a `role: tool` message → model writes the final answer.

With vLLM up on your Spark it talks to that server; otherwise it uses a tool-capable model on this laptop's
Ollama, labelled LAPTOP STAND-IN. The client code is identical either way — that is the point.

Run: .venv/bin/python week25/05_vllm/labs/lab04_tool_calling.py [--model nvidia/Qwen3.6-35B-A3B-NVFP4]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import (banner, chat_any, note, ok, pick_laptop_model, result, show_chat, step,  # noqa: E402
                      warn)

TOOLS = [
    {"type": "function", "function": {
        "name": "get_room_temperature",
        "description": "Current air temperature of a hotel room, in degrees Celsius.",
        "parameters": {"type": "object", "properties": {"room": {"type": "string", "description": "room number"}},
                       "required": ["room"]}}},
    {"type": "function", "function": {
        "name": "set_room_setpoint",
        "description": "Change a hotel room's cooling setpoint, in degrees Celsius (18-28).",
        "parameters": {"type": "object", "properties": {"room": {"type": "string"},
                                                        "celsius": {"type": "number"}},
                       "required": ["room", "celsius"]}}},
]
FAKE_SENSORS = {"1204": 26.5, "1207": 22.0}          # local stand-in data: not a real building


def run_tool(name: str, arguments: str) -> dict:
    try:
        a = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {"error": f"arguments were not JSON: {arguments!r}"}
    if name == "get_room_temperature":
        room = str(a.get("room", ""))
        return {"room": room, "celsius": FAKE_SENSORS.get(room), "source": "fake sensor in lab04"}
    if name == "set_room_setpoint":
        c = float(a.get("celsius", 0))
        return {"room": str(a.get("room")), "setpoint": c, "accepted": 18 <= c <= 28, "source": "dry-run, nothing changed"}
    return {"error": f"unknown tool {name}"}


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="nvidia/Qwen3.6-35B-A3B-NVFP4", help="model id served by vLLM on the Spark")
ap.add_argument("--question", default="Guest in room 1204 says it feels warm. What is the temperature there right now?")
args = ap.parse_args()
laptop = pick_laptop_model(["gemma4:12b", "nemotron-3-nano"])

banner("Lab 05-4 · tool calling through the OpenAI API", "agent-ready model · tool_calls → run tool → final answer")

step(1, "what the client sends: messages + a `tools` list (JSON Schema per function)")
print(f"│ tools: {', '.join(t['function']['name'] for t in TOOLS)}")
print(f"│ user:  {args.question}")
note("vLLM only fills `tool_calls` when started with --enable-auto-tool-choice and a --tool-call-parser that "
     "matches the model family (qwen3_xml for Qwen3.6, qwen3_coder for Nemotron, gemma4 for Gemma 4).")

step(2, "turn 1 — the model decides to call a tool")
msgs = [{"role": "system", "content": "You are a hotel operations assistant. Use tools for live data; never guess."},
        {"role": "user", "content": args.question}]
r1 = chat_any("vllm", args.model, msgs, tools=TOOLS, laptop_model=laptop, max_tokens=150)
show_chat(r1)
calls = r1.get("tool_calls") or []
if r1["source"] == "reference":
    result("DRY: start the agent-ready server on your Spark (TUTORIAL section 4) or Ollama on this laptop.")
    sys.exit(0)
if not calls:
    warn("no tool_calls came back. On vLLM: check --enable-auto-tool-choice and --tool-call-parser. "
         "On the laptop: pick a tool-capable model (gemma4, nemotron-3-nano).")
    sys.exit(0)
ok(f"{len(calls)} tool call(s) — parsed by the server into OpenAI `tool_calls`, not by this script")

step(3, "this script runs the tool locally and sends the result back as role=tool")
msgs.append({"role": "assistant", "content": r1.get("text") or "", "tool_calls": calls})
for i, tc in enumerate(calls):
    fn = tc.get("function") or {}
    out = run_tool(fn.get("name", ""), fn.get("arguments", "{}"))
    print(f"│ {fn.get('name')}({fn.get('arguments')}) → {json.dumps(out)}")
    msgs.append({"role": "tool", "tool_call_id": tc.get("id") or f"call_{i}", "content": json.dumps(out)})

step(4, "turn 2 — the model answers from the tool result")
r2 = chat_any("vllm", args.model, msgs, tools=TOOLS, laptop_model=laptop, max_tokens=150)
show_chat(r2)
if "26.5" in (r2.get("text") or ""):
    ok("the final answer quotes the tool's 26.5 °C — grounded in the tool result, not guessed")
note("Tool calls are sent without streaming (sparkkit turns streaming off when `tools` is set), so the tok/s "
     "above includes prompt processing. Do not use it as a speed benchmark — lab 05-3 is for that.")
result("The server's tool-call parser is what makes a local model usable by OpenAI-style agent frameworks "
       "(NAT in Module 14, OpenClaw and Hermes in Module 17).")
