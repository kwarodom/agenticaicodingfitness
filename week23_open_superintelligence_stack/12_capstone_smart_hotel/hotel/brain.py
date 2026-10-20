#!/usr/bin/env python3
"""The decision-maker  (Week 23 · App 1 Nemotron + App 2 NIM serving).

A Brain decides the next ACT given the running conversation and the tool schemas.
  • NemotronBrain — REAL native tool-calling against an OpenAI-compatible NIM/DGX
    endpoint (the exact `config` connection every Week 23 app uses).
  • SimBrain      — a deterministic, rule-based stand-in so the capstone runs with
    NO GPU ($0). It reasons from the same tool results a model would see.

Same interface, so the *harness* (specialists, policy, relay, flywheel) is identical
in SIM and REAL — swap the brain and the system is production-shaped.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import config


@dataclass
class Decision:
    kind: str                 # "tool" | "final"
    name: str = ""            # tool name
    args: dict = None         # tool args
    text: str = ""            # final answer
    tokens: int = 0           # llm tokens spent on this step


def _called(messages) -> set[str]:
    """tool names already invoked in this conversation."""
    out = set()
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            out.add(tc["function"]["name"])
    return out


def _results_by_tool(messages) -> dict:
    """Map each tool name → its parsed result (pairs assistant tool_call with the next tool msg)."""
    out, pending = {}, None
    for m in messages:
        if m.get("tool_calls"):
            pending = m["tool_calls"][0]["function"]["name"]
        elif m.get("role") == "tool" and pending:
            try:
                out[pending] = json.loads(m["content"])
            except Exception:  # noqa: BLE001
                out[pending] = {}
            pending = None
    return out


class SimBrain:
    """Rule-based stand-in. Faithful to what each specialist role should decide."""

    def decide(self, system, messages, schemas, model, role, ctx) -> Decision:
        room = ctx.get("room", "")
        called = _called(messages)
        res = _results_by_tool(messages)
        tk = 90 if "nano" in model else 150

        if role == "maintenance":
            tele = res.get("get_room_telemetry", {})
            if "get_room_telemetry" not in called:
                return Decision("tool", "get_room_telemetry", {"room": room}, tokens=tk)
            if tele.get("occupied") and tele.get("delta_c", 0) > 3 and "search_sop" not in called:
                return Decision("tool", "search_sop",
                                {"query": "critical occupied room above setpoint"}, tokens=tk)
            if "dispatch_work_order" not in called:
                return Decision("tool", "dispatch_work_order",
                                {"room": room, "priority": "CRITICAL"}, tokens=tk)
            return Decision("final",
                            text=f"Room {room}: occupied and >3°C above setpoint → CRITICAL. "
                                 f"Dispatched a CRITICAL work order (per SOP).", tokens=tk)

        if role == "energy":
            tele = res.get("get_room_telemetry", {})
            if "get_room_telemetry" not in called:
                return Decision("tool", "get_room_telemetry", {"room": room}, tokens=tk)
            occupied = tele.get("occupied", True)
            if not occupied and "set_setpoint" not in called:
                # trim an empty room up toward the policy ceiling to save energy
                return Decision("tool", "set_setpoint", {"room": room, "setpoint_c": 25.0}, tokens=tk)
            if occupied and "search_sop" not in called:
                return Decision("tool", "search_sop", {"query": "energy trim unoccupied rooms first"}, tokens=tk)
            saved = res.get("set_setpoint", {}).get("est_kw_saved", 0)
            return Decision("final",
                            text=(f"Room {room} is empty — raised setpoint to 25°C, saved ~{saved} kW."
                                  if not occupied else
                                  f"Room {room} is occupied — held comfort per SOP; deferred trim."), tokens=tk)

        if role == "guest":
            prof = res.get("guest_profile", {})
            if "guest_profile" not in called:
                return Decision("tool", "guest_profile", {"room": room}, tokens=tk)
            if prof.get("vip") and "search_sop" not in called:
                return Decision("tool", "search_sop", {"query": "vip comfort approval"}, tokens=tk)
            if prof.get("vip") and "set_setpoint" not in called:
                # attempt a comfort setpoint — the policy will require human approval for a VIP room
                return Decision("tool", "set_setpoint", {"room": room, "setpoint_c": 22.5}, tokens=tk)
            if prof.get("vip"):
                return Decision("final",
                                text=f"Room {room}: VIP — autonomous setpoint change blocked by policy; "
                                     f"routed to a human concierge for approval.", tokens=tk)
            return Decision("final", text=f"Room {room}: adjusted for comfort within policy.", tokens=tk)

        return Decision("final", text="No action.", tokens=tk)


class NemotronBrain:
    """REAL native tool-calling against the connected Nemotron NIM/DGX endpoint."""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=180.0)

    def decide(self, system, messages, schemas, model, role, ctx) -> Decision:
        msgs = [{"role": "system", "content": system}] + messages
        r = self.client.chat.completions.create(
            model=model, messages=msgs, tools=schemas, max_tokens=400, temperature=0.2)
        msg = r.choices[0].message
        tokens = getattr(getattr(r, "usage", None), "completion_tokens", 0) or 120
        calls = msg.tool_calls or []
        if not calls:
            return Decision("final", text=(msg.content or "").strip(), tokens=tokens)
        c = calls[0]
        try:
            args = json.loads(c.function.arguments or "{}")
        except Exception:  # noqa: BLE001
            args = {}
        return Decision("tool", c.function.name, args, tokens=tokens)


def make_brain():
    """REAL if an endpoint is reachable (config.MODE=='real'), else the SIM stand-in."""
    if config.MODE == "real":
        try:
            return NemotronBrain(), "real"
        except Exception:  # noqa: BLE001
            pass
    return SimBrain(), "sim"
