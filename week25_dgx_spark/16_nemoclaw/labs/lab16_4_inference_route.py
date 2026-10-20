#!/usr/bin/env python3
"""Lab 16-4 · Inference route check: will the model endpoint NemoClaw routes to work for an agent?

NemoClaw's "Existing vLLM" option (NEMOCLAW_PROVIDER=vllm) expects a ready server on the
Spark's localhost:8000; inside the sandbox the agent reaches it as `inference.local`.
An agent needs three things from that endpoint, so this lab checks three things:

  1. GET /v1/models lists a model,
  2. a short chat answers ("Reply with exactly: READY" — the playbook's own smoke test),
  3. a chat with one tool returns a well-formed tool call (name + JSON arguments).

It runs against vLLM on your Spark when :8000 answers. Without a Spark it runs for
REAL against Ollama on this laptop, labelled LAPTOP STAND-IN: the API shape is the
same, the speed is not the Spark's.

Run: .venv/bin/python week25/16_nemoclaw/labs/lab16_4_inference_route.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import (banner, chat_any, check, models, note, pick_laptop_model, resolve, result,  # noqa: E402
                      show_chat, step, table)

TOOLS = [{"type": "function", "function": {
    "name": "get_sandbox_status",
    "description": "Return the status of a NemoClaw sandbox by name.",
    "parameters": {"type": "object", "properties": {"sandbox": {"type": "string", "description": "sandbox name"}},
                   "required": ["sandbox"]}}}]

banner("Lab 16-4 · inference route check", "models · a plain answer · a tool call — what an always-on agent needs")

base, src = resolve("vllm")
where_s = {"spark": "vLLM on your Spark (:8000)", "laptop": "Ollama on THIS laptop — LAPTOP STAND-IN, not the Spark",
           "dry": "nothing reachable (DRY)"}[src]
note(f"endpoint: {base or '—'} · {where_s}")

step(1, "GET /v1/models")
ids = models(base) if base else []
print(f"→ GET {base or '<endpoint>'}/models")
print("· " + (", ".join(ids[:6]) + (" …" if len(ids) > 6 else "") if ids else "(no models)"))
model = ids[0] if src == "spark" and ids else ""
laptop_model = pick_laptop_model(["gemma4:12b", "nemotron-3-nano"]) if src == "laptop" else None

step(2, "a plain answer (the playbook's smoke test prompt)")
r1 = chat_any("vllm", model or "default", [{"role": "user", "content": "Reply with exactly: READY"}],
              max_tokens=256, laptop_model=laptop_model, reference="READY")
show_chat(r1)
said_ready = "READY" in (r1.get("text") or "").upper()

step(3, "one tool, one question — does the model emit a tool call?")
r2 = chat_any("vllm", model or "default",
              [{"role": "system", "content": "You manage NemoClaw sandboxes. Use tools when they help."},
               {"role": "user", "content": "Is the sandbox named my-assistant healthy?"}],
              max_tokens=256, tools=TOOLS, laptop_model=laptop_model)
show_chat(r2)
calls = r2.get("tool_calls") or []
fn = (calls[0].get("function") or {}) if calls else {}
try:
    args = json.loads(fn.get("arguments") or "{}")
except json.JSONDecodeError:
    args = None
well_formed = fn.get("name") == "get_sandbox_status" and isinstance(args, dict) and "sandbox" in args

real = r2.get("source") in ("spark", "laptop")
print()
table([["models listed", "✓" if ids else "✕", f"{len(ids)} model(s)"],
       ["plain answer", "✓" if said_ready else "✕", (r1.get("text") or "—")[:40]],
       ["tool call", "✓" if well_formed else "✕", f"{fn.get('name')}({fn.get('arguments')})" if fn else "none"]],
      ["check", "result", "detail"])
print()
if real:
    ok = check(bool(ids) and said_ready and well_formed,
               f"this endpoint can drive an agent ({'Spark' if src == 'spark' else 'LAPTOP STAND-IN'})",
               "this endpoint is not agent-ready — see the ✕ rows")
    if src == "spark" and ok:
        note("Onboard with it: NEMOCLAW_PROVIDER=vllm (the 'Existing vLLM' choice). Inside the sandbox it is inference.local.")
    elif src == "laptop":
        note("Laptop result: this proves the client side (request shape, tool_calls parsing). On the Spark, start vLLM "
             "with tool calling (Module 05) or let Express Install manage it, then re-run.")
    result("A model that lists and chats but never emits tool_calls gives you a chatbot, not an agent: "
           "OpenClaw's tools and skills would never fire.")
else:
    result("DRY: no endpoint reachable. Start Ollama on this laptop or vLLM on the Spark and run again.")
