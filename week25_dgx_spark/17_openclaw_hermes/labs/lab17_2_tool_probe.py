#!/usr/bin/env python3
"""Lab 17-2 · Tool-calling probe: can this model endpoint really drive OpenClaw or Hermes?

Both agents send OpenAI-style tool definitions to the model and run whatever tool calls come
back. So before you install either one, probe the endpoint with five small tests:

  P1  call      — asked something only a tool can answer, does it emit a tool call?
  P2  choose    — with three tools offered, does it pick the right one?
  P3  schema    — are the arguments valid JSON with the required keys and right types?
  P4  round trip— given the tool's result, does the final answer use it?
  P5  restraint — asked "2 + 2", does it answer directly instead of calling a tool?

On the Spark it probes vLLM on :8000. Without a Spark it probes Ollama on THIS laptop for
real, labelled LAPTOP STAND-IN, one model after another (small requests; ~5 calls per model).

Run: .venv/bin/python week25/17_openclaw_hermes/labs/lab17_2_tool_probe.py [--models gemma4:12b,gemma3:4b]
"""
import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, chat, laptop_models, models, note, resolve, result, step, table, warn  # noqa: E402


def fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


TOOLS = [
    fn("get_time", "Current local time in a city.", {"city": {"type": "string"}}, ["city"]),
    fn("convert_currency", "Convert an amount between two ISO currency codes.",
       {"amount": {"type": "number"}, "from_ccy": {"type": "string"}, "to_ccy": {"type": "string"}},
       ["amount", "from_ccy", "to_ccy"]),
    fn("lookup_ticket", "Look up a support ticket by its id.", {"ticket_id": {"type": "string"}}, ["ticket_id"]),
]
SYSTEM = {"role": "system", "content": "You are an assistant with tools. Use a tool only when it is needed."}
TOOL_RESULT = {"amount": 120, "from_ccy": "USD", "to_ccy": "THB", "result": 4380.0}


def call(base: str, model: str, messages: list, laptop: bool, tools=TOOLS) -> tuple[dict | None, str]:
    extra = {"reasoning_effort": "none"} if laptop else None
    try:
        return chat(base, model, messages, tools=tools, max_tokens=200, temperature=0, extra=extra, timeout=120), ""
    except HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            body = json.loads(body).get("error", {}).get("message", body)
        except (json.JSONDecodeError, AttributeError):
            pass
        return None, f"HTTP {e.code}: {str(body)[:70]}"
    except (URLError, TimeoutError, OSError) as e:
        return None, f"{type(e).__name__}: {str(e)[:70]}"


def first_call(r: dict | None) -> tuple[str, dict | None]:
    tcs = (r or {}).get("tool_calls") or []
    if not tcs:
        return "", None
    f = tcs[0].get("function") or {}
    try:
        args = json.loads(f.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = None
    return f.get("name") or "", args


def probe(base: str, model: str, laptop: bool) -> tuple[list[str], str, float]:
    """Return ([P1..P5 marks], note, seconds)."""
    t0 = time.perf_counter()
    user = {"role": "user", "content": "Convert 120 USD to THB."}
    r, err = call(base, model, [SYSTEM, user], laptop)
    if r is None:
        return ["✕"] * 5, err, time.perf_counter() - t0
    name, args = first_call(r)
    p1 = bool(name)
    p2 = name == "convert_currency"
    p3 = (isinstance(args, dict) and all(k in args for k in ("amount", "from_ccy", "to_ccy"))
          and isinstance(args.get("amount"), (int, float)) and str(args.get("to_ccy", "")).upper() == "THB")
    p4 = False
    if p2:
        tc = r["tool_calls"][0]
        msgs = [SYSTEM, user,
                {"role": "assistant", "content": "", "tool_calls": [{"id": tc.get("id") or "call_1", "type": "function",
                                                                     "function": tc["function"]}]},
                {"role": "tool", "tool_call_id": tc.get("id") or "call_1", "content": json.dumps(TOOL_RESULT)}]
        r4, _ = call(base, model, msgs, laptop)
        text = ((r4 or {}).get("text") or "").replace(",", "").replace(" ", "")
        p4 = "4380" in text
    r5, _ = call(base, model, [SYSTEM, {"role": "user", "content": "What is 2 + 2? Answer with just the number."}], laptop)
    p5 = r5 is not None and not first_call(r5)[0] and "4" in (r5.get("text") or "")
    detail = f"{name}({json.dumps(args, separators=(',', ':')) if args is not None else '?'})" if name else "no tool call"
    return ["✓" if x else "✕" for x in (p1, p2, p3, p4, p5)], detail, time.perf_counter() - t0


ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--models", default="", help="comma-separated laptop models to probe (stand-in only)")
args = ap.parse_args()

banner("Lab 17-2 · tool-calling probe", "five small tests an agent-ready endpoint must pass")
base, src = resolve("vllm")
if src == "dry":
    warn("No endpoint reachable (no Spark vLLM on :8000, no laptop Ollama). Start one and run again.")
    result("DRY: this lab only prints real answers, so there is nothing to show without an endpoint.")
    sys.exit(0)

if src == "spark":
    targets = models(base)[:1]
    note(f"probing vLLM on your Spark · {base} · {targets[0] if targets else '(no model listed)'}")
else:
    have = [m for m in laptop_models() if "cloud" not in m]
    want = [m.strip() for m in args.models.split(",") if m.strip()] or ["gemma4:12b", "nemotron-3-nano:latest", "gemma3:4b"]
    targets = [m for m in want if m in have] or have[:1]
    note(f"probing Ollama on THIS laptop · {base} · LAPTOP STAND-IN — tool behaviour is the model's, speed is not the Spark's")

rows = []
for i, model in enumerate(targets, 1):
    step(i, f"probe {model}")
    marks, detail, secs = probe(base, model, laptop=(src == "laptop"))
    print(f"│ P1 call {marks[0]} · P2 choose {marks[1]} · P3 schema {marks[2]} · P4 round trip {marks[3]} · "
          f"P5 restraint {marks[4]}")
    print(f"· {'first tool call' if marks[0] == '✓' else 'why'}: {detail}")
    verdict = "agent-ready" if all(m == "✓" for m in marks[:4]) else ("chat only" if marks[0] == "✕" else "unreliable")
    rows.append([model, *marks, verdict, f"{secs:.0f}s"])

print()
table(rows, ["model", "P1", "P2", "P3", "P4", "P5", "verdict", "wall time"])
note("P1–P4 must pass for OpenClaw/Hermes to use tools. P5 failing means the agent calls tools when it should not "
     "(slower, and riskier once the tools can run commands).")
if src == "laptop":
    note("LAPTOP STAND-IN: these rows describe the laptop's models, not the Spark's. On the Spark, run this again "
         "against nvidia/Qwen3.6-35B-A3B-NVFP4 served with the agent-ready recipe.")
result("Pick the model for OpenClaw's `id` / Hermes' `model.default` from the agent-ready rows only.")
