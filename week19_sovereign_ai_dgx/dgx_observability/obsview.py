#!/usr/bin/env python3
"""Make a **sovereign agent observable** — run it and trace every step.

Runs a small tool-calling agent (a smart-hotel HVAC triage agent) against the
DGX model and wraps each step — the agent, every LLM call, every tool call — in a
span via tracer.py. In REAL mode the agent makes genuine local calls; in SIM it's
a scripted loop. Either way you get the same Phoenix-shaped span tree to inspect.

Output is PLAIN text so it renders in a terminal and the web app.
"""
from __future__ import annotations

import json
import sys
import time
from urllib.parse import urlparse

import config
import tracer as T

S = "▣"


def _p(line: str = "") -> None:
    print(line, flush=True)


def is_sim() -> bool:
    return config.MODE != "real"


def banner(part: str, title: str, level: str) -> None:
    _p("━" * 64)
    _p(f"  {part} — {title}   [{level}]")
    _p("━" * 64)
    _p("")


def mode_line() -> None:
    if is_sim():
        _p(f"{S} MODE: SIM — agent loop simulated; spans rendered from the simulation.")
        _p(f"  connection: {config.CONN} ({config.conn_human()}) — nothing reachable yet.")
    else:
        _p(f"{S} MODE: REAL · connection = {config.CONN} ({config.conn_human()}).")
        _p(f"  agent calls {config.MODEL} @ {config.safe_base_url()}.")
    px = "up ✓" if config.phoenix_up() else "not running (we render the tree locally)"
    _p(f"  Phoenix at {config.PHOENIX_ENDPOINT}: {px}")
    _p("")


def _client():
    from openai import OpenAI
    return OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=120.0)


def require_runtime() -> bool:
    return True


# ── the demo agent's tools (smart-hotel HVAC triage) ──────────────────────────
ROOMS = {
    "1203": {"temp_c": 26.4, "setpoint_c": 22.0, "filter_pa": 120, "occupied": True},
    "0907": {"temp_c": 22.1, "setpoint_c": 22.0, "filter_pa": 265, "occupied": False},
}

TOOLS = [
    {"type": "function", "function": {
        "name": "get_room_telemetry",
        "description": "Read live HVAC telemetry for a hotel room.",
        "parameters": {"type": "object", "properties": {
            "room": {"type": "string"}}, "required": ["room"]}}},
    {"type": "function", "function": {
        "name": "dispatch_work_order",
        "description": "Dispatch maintenance for a room at a given priority.",
        "parameters": {"type": "object", "properties": {
            "room": {"type": "string"},
            "priority": {"type": "string", "enum": ["CRITICAL", "ROUTINE"]},
            "reason": {"type": "string"}}, "required": ["room", "priority", "reason"]}}},
]


def _impl_get(args: dict) -> str:
    r = ROOMS.get(str(args.get("room")), {})
    return json.dumps(r) if r else "{}"


def _impl_dispatch(args: dict) -> str:
    return json.dumps({"work_order": "WO-" + str(args.get("room")),
                       "priority": args.get("priority"), "status": "dispatched"})


IMPLS = {"get_room_telemetry": _impl_get, "dispatch_work_order": _impl_dispatch}


def _toks(text: str) -> int:
    return max(1, round(len(text) / 4))


def traced_agent_run(task: str, *, project: str = "sovereign-agent") -> T.Tracer:
    """Run the HVAC triage agent on the DGX model, tracing every span."""
    tr = T.Tracer(project=project)
    with tr.span("hvac_triage_agent", T.AGENT, **{"input.value": task}) as agent:
        if is_sim():
            _simulated_loop(tr, task)
        else:
            _real_loop(tr, task)
        agent.attributes["output.value"] = "alarm queue cleared"
    return tr


def _simulated_loop(tr: T.Tracer, task: str) -> None:
    # Round 1: LLM decides to read telemetry for the hot room.
    with tr.span("chat.completions", T.LLM,
                 **{"gen_ai.request.model": config.MODEL,
                    "gen_ai.usage.input_tokens": _toks(task) + 80,
                    "gen_ai.usage.output_tokens": 24}):
        time.sleep(0.05)
    with tr.span("get_room_telemetry", T.TOOL,
                 **{"tool.name": "get_room_telemetry",
                    "input.value": '{"room":"1203"}',
                    "output.value": _impl_get({"room": "1203"})}):
        time.sleep(0.01)
    # Round 2: LLM reasons it's guest-impacting → dispatch CRITICAL.
    with tr.span("chat.completions", T.LLM,
                 **{"gen_ai.request.model": config.MODEL,
                    "gen_ai.usage.input_tokens": 150,
                    "gen_ai.usage.output_tokens": 40}):
        time.sleep(0.05)
    with tr.span("dispatch_work_order", T.TOOL,
                 **{"tool.name": "dispatch_work_order",
                    "input.value": '{"room":"1203","priority":"CRITICAL"}',
                    "output.value": _impl_dispatch({"room": "1203", "priority": "CRITICAL"})}):
        time.sleep(0.01)
    with tr.span("chat.completions", T.LLM,
                 **{"gen_ai.request.model": config.MODEL,
                    "gen_ai.usage.input_tokens": 200,
                    "gen_ai.usage.output_tokens": 60}):
        time.sleep(0.05)


def _real_loop(tr: T.Tracer, task: str) -> None:
    client = _client()
    messages = [
        {"role": "system", "content": "You are a smart-hotel HVAC triage agent. Use the "
         "tools to read telemetry and dispatch maintenance. A guest-occupied room far "
         "from setpoint is CRITICAL; a high filter ΔP on an empty room is ROUTINE."},
        {"role": "user", "content": task},
    ]
    for _ in range(5):
        with tr.span("chat.completions", T.LLM,
                     **{"gen_ai.request.model": config.MODEL,
                        "gen_ai.system": "openai"}) as span:
            resp = client.chat.completions.create(
                model=config.MODEL, messages=messages, tools=TOOLS,
                max_tokens=config.DEFAULT_MAX_TOKENS, temperature=0.2)
            msg = resp.choices[0].message
            u = getattr(resp, "usage", None)
            span.attributes["gen_ai.usage.input_tokens"] = getattr(u, "prompt_tokens", 0) or 0
            span.attributes["gen_ai.usage.output_tokens"] = getattr(u, "completion_tokens", 0) or 0
        calls = msg.tool_calls or []
        if not calls:
            break
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name,
                                                      "arguments": c.function.arguments}}
                                        for c in calls]})
        for c in calls:
            args = json.loads(c.function.arguments or "{}")
            with tr.span(c.function.name, T.TOOL,
                         **{"tool.name": c.function.name,
                            "input.value": c.function.arguments}) as span:
                result = IMPLS.get(c.function.name, lambda a: "{}")(args)
                span.attributes["output.value"] = result
            messages.append({"role": "tool", "tool_call_id": c.id, "content": result})


def show_tree(tr: T.Tracer) -> None:
    _p(tr.render_tree())
    t = tr.totals()
    _p("")
    _p(f"  ◆ {t['spans']} spans · {t['llm_calls']} LLM calls · {t['tool_calls']} tool calls "
       f"· {t['total_tokens']} tokens · {t['latency_ms']:.0f} ms · {t['errors']} errors · $0")


if __name__ == "__main__":
    _p("obsview.py is a helper imported by the demos in demos/.")
    sys.exit(0)
