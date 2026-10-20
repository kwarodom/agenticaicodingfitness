#!/usr/bin/env python3
"""Specialized agents + orchestration.

  • Specialist   — a NemoClaw-built domain agent (Week 23 · App 6): persona + base
    model + skills + tools + signed policy, running a bounded ReAct loop. Every tool
    call goes through the OpenShell Gateway and is observed by NeMo Relay.
  • Orchestrator — the AI-Q Open Agent Blueprint (Week 23 · App 5): an Intent Router
    (Nemotron Nano) that routes/escalates, and a Deep Agent (Nemotron Super) that plans
    and fans work out to the specialist sub-agents.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import relay as relay_mod
from .policy import Gateway, signature_line


@dataclass
class Spec:
    """A NemoClaw specialized-agent spec."""
    role: str
    name: str
    persona: str
    skills: list = field(default_factory=list)


SPECS = {
    "energy": Spec("energy", "Energy·HVAC agent",
                   "You optimize energy without dropping guest comfort below SOP thresholds.",
                   ["NeMo Retriever RAG (SOP)", "cuOpt scheduling"]),
    "maintenance": Spec("maintenance", "Maintenance agent",
                        "You triage HVAC faults and dispatch work orders by SOP priority.",
                        ["NeMo Retriever RAG (SOP)"]),
    "guest": Spec("guest", "Guest-Experience agent",
                  "You protect guest comfort; VIP rooms need human approval.",
                  ["NeMo Retriever RAG (SOP)"]),
}


@dataclass
class Result:
    role: str
    answer: str
    actions: list          # tool calls that actually ran
    denials: list          # policy denials encountered
    needs_human: bool = False


class Specialist:
    MAX_STEPS = 5

    def __init__(self, spec: Spec, brain, tools, gateway: Gateway, relay: relay_mod.Relay):
        self.spec, self.brain, self.tools = spec, brain, tools
        self.gateway, self.relay = gateway, relay

    def _sys(self) -> str:
        return (f"You are the {self.spec.name} for a hotel. {self.spec.persona} "
                f"Use the tools; pass room ids bare (e.g. '1203'). Skills: {', '.join(self.spec.skills)}.")

    def handle(self, task: str, room: str, difficulty: str) -> Result:
        model = self.relay.route(difficulty)
        messages = [{"role": "user", "content": task}]
        actions, denials, needs_human = [], [], False
        self.relay.span("agent", self.spec.name, depth=1, detail=f"model={model.split(':')[0]}")

        for step in range(self.MAX_STEPS):
            d = self.brain.decide(self._sys(), messages, self.tools.SCHEMAS, model,
                                  self.spec.role, {"room": room})
            self.relay.span("llm", "decide", model=model, tokens=d.tokens, ms=120, depth=2)
            if d.kind == "final":
                return Result(self.spec.role, d.text, actions, denials, needs_human)

            call_id = f"c{step}"
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": call_id, "type": "function",
                 "function": {"name": d.name, "arguments": json.dumps(d.args or {})}}]})

            verdict = self.gateway.check(self.spec.role, d.name, d.args or {})
            if not verdict.allow:
                needs_human = needs_human or verdict.needs_human
                denials.append(verdict.reason)
                res = json.dumps({"policy_denied": verdict.reason, "needs_human": verdict.needs_human})
                self.relay.span("tool", f"{d.name} ⛔DENIED", ok=False, ms=2, depth=2, detail=verdict.reason)
            else:
                res = self.tools.run(d.name, d.args or {})
                short = res if len(res) < 60 else res[:57] + "…"
                self.relay.span("tool", d.name, ms=8, depth=2, detail=short)
                actions.append({"tool": d.name, "args": d.args, "result": json.loads(res) if res else {}})
            messages.append({"role": "tool", "tool_call_id": call_id, "content": res})

        return Result(self.spec.role, "(reached step limit)", actions, denials, needs_human)


class Orchestrator:
    """AI-Q: Intent Router (Nano) → Deep Agent (Super) → specialist sub-agents."""

    def __init__(self, state, tools, gateway, relay, brain):
        self.state, self.tools, self.gateway, self.relay = state, tools, gateway, relay
        self.specialists = {r: Specialist(SPECS[r], brain, tools, gateway, relay) for r in SPECS}

    # Intent Router (Nemotron Nano, NAT-optimized) — cheap classify + route.
    def route(self, text: str) -> tuple[str, str]:
        t = text.lower()
        if any(w in t for w in ("alarm", "fault", "broken", "critical", "hot", "leak")):
            return "maintenance", "hard"      # triage → escalate to Super
        if any(w in t for w in ("energy", "kw", "load", "save", "empty", "unoccupied")):
            return "energy", "simple"
        return "guest", "simple"

    def handle_event(self, text: str, room: str) -> Result:
        role, difficulty = self.route(text)
        self.relay.span("agent", "Intent Router", model=relay_mod.NANO, tokens=40, ms=30,
                        detail=f"→ {role} ({difficulty})")
        return self.specialists[role].handle(text, room, difficulty)

    # Deep Agent (Nemotron Super) — plan the morning sweep, fan out to sub-agents.
    def morning_brief(self) -> dict:
        self.relay.span("agent", "Deep Agent · plan morning sweep", model=relay_mod.SUPER,
                        tokens=180, ms=210, detail="decompose → delegate")
        results = []
        for rid, r in self.state.rooms.items():
            if r.id == "lobby":
                continue
            if r.occupied and (r.temp_c - r.setpoint_c) > 3:
                results.append(("maintenance", rid,
                                self.specialists["maintenance"].handle(
                                    f"Room {rid} alarm — triage and act.", rid, "hard")))
            elif not r.occupied:
                results.append(("energy", rid,
                                self.specialists["energy"].handle(
                                    f"Room {rid} is empty — reduce energy if safe.", rid, "simple")))
            elif r.vip:
                results.append(("guest", rid,
                                self.specialists["guest"].handle(
                                    f"VIP in room {rid} requested it a little cooler.", rid, "simple")))
        return {"summary": self.state.summary(), "results": results,
                "policy": signature_line()}
