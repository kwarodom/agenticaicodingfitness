#!/usr/bin/env python3
"""Lab 03-4 · Tool calling through Ollama's OpenAI API: the model asks, your code answers.

Gives the model two real tools — the memory and speed formulas from Module 01 (sparkkit's
weights_gb and decode_ceiling_tok_s) — and asks a question it can only answer by calling
them. The full loop runs for real: request with `tools` → the model returns `tool_calls` →
this script runs the functions → it sends the results back as `role: tool` messages → the
model writes the final answer. Uses the Spark's Ollama when it answers, otherwise the
laptop stand-in (labelled). Checks /api/show first: only models with the "tools"
capability can do this.

Run: .venv/bin/python week25/03_ollama_open_webui/labs/lab04_tool_calling.py
"""
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import (banner, chat, decode_ceiling_tok_s, http_json, note, pick_laptop_model, resolve,  # noqa: E402
                      result, show_chat, step, warn, weights_gb)

SPARK_MODEL = "qwen3.6:35b-a3b"          # the Open WebUI playbook's agent-ready model for DGX Spark
FORMATS = ["bf16", "fp8", "mxfp4", "nvfp4", "q8_0", "q4_k_m"]
TOOLS = [
    {"type": "function", "function": {
        "name": "weights_gb",
        "description": "Size in GB of a model's weights at a given number format.",
        "parameters": {"type": "object", "properties": {
            "params_b": {"type": "number", "description": "total parameters, in billions"},
            "fmt": {"type": "string", "enum": FORMATS}}, "required": ["params_b", "fmt"]}}},
    {"type": "function", "function": {
        "name": "decode_ceiling_tok_s",
        "description": "Upper bound on single-stream tokens/second on a DGX Spark (273 GB/s memory bandwidth).",
        "parameters": {"type": "object", "properties": {
            "active_params_b": {"type": "number", "description": "parameters read per token, in billions"},
            "fmt": {"type": "string", "enum": FORMATS}}, "required": ["active_params_b", "fmt"]}}},
]
ERRORS = (HTTPError, URLError, TimeoutError, OSError)


def stop(what: str, e: Exception) -> None:
    """A busy or failing server ends the lab with a clear line, never a traceback."""
    warn(f"{what} failed: {e} — the server may be busy loading another model. Run the lab again later.")
    sys.exit(0)


IMPL = {"weights_gb": lambda a: round(weights_gb(float(a["params_b"]), a["fmt"]), 1),
        "decode_ceiling_tok_s": lambda a: round(decode_ceiling_tok_s(float(a["active_params_b"]), a["fmt"]), 1)}
QUESTION = ("gpt-oss-120b has 117B total parameters and 5.1B active per token, stored in mxfp4. "
            "Use the tools: how big are its weights, and what is the decode ceiling on a DGX Spark?")

base, src = resolve("ollama")
if src == "dry":
    banner("Lab 03-4 · tool calling", "no Ollama reachable on the Spark or this laptop")
    print("◈ DRY — start Ollama on the Spark (Section 1) or on this laptop, then run again.")
    sys.exit(0)
model = SPARK_MODEL if src == "spark" else pick_laptop_model(["nemotron-3-nano"])
who = "LIVE on the Spark" if src == "spark" else "LAPTOP STAND-IN (Ollama on this laptop, not the Spark)"
banner("Lab 03-4 · tool calling through /v1/chat/completions", f"{model} · {who}")
extra = {"reasoning_effort": "none"}         # answer directly; thinking is not needed to pick a tool

step(1, "can this model call tools? (POST /api/show → capabilities)")
try:
    caps = http_json("POST", base[:-3] + "/api/show", {"model": model}, timeout=30).get("capabilities", [])
except ERRORS as e:
    stop("POST /api/show", e)
print(f"◆ {model}: {', '.join(caps)}")
if "tools" not in caps:
    warn("no 'tools' capability — Ollama will reject the request. Pick a model whose page lists 'tools'.")
    sys.exit(0)

step(2, "ask, with two tools on offer")
messages = [{"role": "user", "content": QUESTION}]
print(f"→ POST {base}/chat/completions · tools=[{', '.join(t['function']['name'] for t in TOOLS)}]")
try:
    r = chat(base, model, messages, tools=TOOLS, max_tokens=200, extra=extra)
except ERRORS as e:
    stop("the first chat request", e)
r["source"] = src
show_chat(r)

for turn in range(3):                            # a model may need more than one round of calls
    calls = r.get("tool_calls") or []
    if not calls:
        break
    step(3 + turn, f"run {len(calls)} tool call(s) here, send the results back as role=tool")
    messages.append({"role": "assistant", "content": r.get("text") or "", "tool_calls": calls})
    for tc in calls:
        fn = tc.get("function") or {}
        raw = fn.get("arguments") or "{}"
        try:
            args = raw if isinstance(raw, dict) else json.loads(raw)
            out = IMPL[fn["name"]](args)
        except Exception as e:  # noqa: BLE001 — a bad call is data for the model, not a crash
            out = f"error: {type(e).__name__}: {e}"
        print(f"│ {fn.get('name')}({raw if isinstance(raw, str) else json.dumps(raw)}) → {out}")
        messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "name": fn.get("name"),
                         "content": json.dumps(out)})
    try:
        r = chat(base, model, messages, tools=TOOLS, max_tokens=200, extra=extra)
    except ERRORS as e:
        stop("the follow-up request with the tool results", e)
    r["source"] = src
    show_chat(r)

if not r.get("tool_calls") and r.get("text"):
    ok_w, ok_c = weights_gb(117, "mxfp4"), decode_ceiling_tok_s(5.1, "mxfp4")
    note(f"ground truth from sparkkit: weights {ok_w:.1f} GB · ceiling {ok_c:.1f} tok/s — "
         "does the final answer quote these numbers?")
note("The model never runs code. It returns a name + JSON arguments; YOUR program decides whether to run it. "
     "That gap is where Modules 15–16 put the sandbox and the policy.")
result(f"tool calling = tools in the request, tool_calls in the reply, role=tool messages back. Source: {who}.")
