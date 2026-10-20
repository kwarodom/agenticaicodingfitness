#!/usr/bin/env python3
"""CHECKPOINT 5 — A2A: agent cards and the cross-framework handoff.

Goal: delegate across the boundary CP4 ran into. The parts vendor runs its OWN
agent on its OWN framework. A2A (Agent-to-Agent, Google → Linux Foundation) is
the cross-vendor protocol that lets your coordinator DISCOVER it, AUTHENTICATE,
and hand it a task — without importing its code, the way A2A is to agents what
MCP is to tools.

Two halves of the protocol:
  Part A — the Agent Card: a public JSON manifest at /.well-known/agent.json that
           says name / url / authSchemes / skills[] (each skill has an
           input_schema). The caller reads it to decide "is this the specialist
           I need, can I auth, what input does its skill want?"
  Part B — the task lifecycle: delegation is a TASK with six states, not a
           function call:
             submitted → working → completed                       (happy path)
                            ├→ input-required → (resume) → working →┘
                            ├→ failed
                            └→ canceled
           `input-required` is human-in-the-loop ACROSS A NETWORK: the vendor
           pauses to ask a question, your side answers, the task resumes.

This is an offline MOCK (no network, no key) so the PROTOCOL is what's under
test, not a model. The real shape (a2a-sdk / Google ADK's RemoteA2aAgent) is
shown at the bottom as guarded, illustrative code.

(Week 17 · fleet orchestration · A2A skill: cards + six-state lifecycle)

Run:  python week17/checkpoints/checkpoint5_a2a_cards.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # config.py
from config import MockLLM, banner, have_a2a, step


# ════════════════════════════════════════════════════════════════════════════
# Part A — the Agent Card the VENDOR publishes at /.well-known/agent.json
# (spec v0.3+ serves it at /.well-known/agent-card.json; agent.json is legacy).
# ════════════════════════════════════════════════════════════════════════════
VENDOR_CARD = {
    "schemaVersion": "1.0",
    "humanReadableId": "hvac-parts/fulfillment",
    "name": "Acme HVAC Parts Supplier",
    "description": "Quotes, reserves, and ships HVAC replacement parts.",
    "url": "https://parts.acme-hvac.example.com/a2a",
    "capabilities": {"a2aVersion": "1.0", "supportsPushNotifications": True},
    "authSchemes": [{"scheme": "oauth2", "tokenUrl": "https://auth.acme-hvac.example.com/token"}],
    "skills": [
        {
            "id": "part-fulfillment",
            "name": "Part Fulfillment",
            "description": "Reserve and ship a part given its SKU and ship-to dock.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string"},
                    "room": {"type": "string"},
                    "urgency": {"type": "string", "enum": ["standard", "rush"]},
                },
                "required": ["sku"],
            },
        }
    ],
}

REQUIRED_CARD_FIELDS = ["name", "url", "authSchemes", "skills"]


def validate_card(card: dict) -> bool:
    """The discovery-side check the CALLER runs before delegating anything."""
    missing = [f for f in REQUIRED_CARD_FIELDS if f not in card]
    assert not missing, f"card missing required field(s): {missing}"
    assert card["authSchemes"], "card advertises no authSchemes — cannot authenticate"
    for s in card["skills"]:
        assert {"id", "input_schema"} <= s.keys(), f"skill {s} missing id/input_schema"
    return True


# ════════════════════════════════════════════════════════════════════════════
# Part B — the mock vendor agent: walks the six-state task lifecycle, including
# an `input-required` pause (it needs to know which loading dock to ship to).
# ════════════════════════════════════════════════════════════════════════════
class MockVendorAgent:
    """Stand-in for a remote A2A agent. Returns task dicts whose `state` walks
    the lifecycle. It pauses (input-required) once to ask a clarifying question,
    then completes when the caller supplies the answer."""

    def submit(self, skill_id: str, inputs: dict) -> dict:
        if skill_id != "part-fulfillment":
            return {"state": "failed", "reason": f"unknown skill {skill_id!r}"}
        if "sku" not in inputs:
            return {"state": "failed", "reason": "input_schema requires 'sku'"}
        # vendor needs more info before it can ship → HITL across the network
        return {
            "state": "input-required",
            "task_id": "task-7781",
            "question": "Which loading dock should we ship to (A or B)?",
        }

    def resume(self, task_id: str, answer: str) -> dict:
        return {
            "state": "completed",
            "task_id": task_id,
            "result": {"eta_days": 2, "tracking_id": "1Z999",
                       "ship_to_dock": answer, "status": "reserved+shipped"},
        }


def orchestrator_delegate(card: dict, user_msg: str, llm) -> dict:
    """Your coordinator delegating a part order to the vendor over A2A."""
    step("1) DISCOVER — fetch + validate the vendor's Agent Card")
    validate_card(card)
    step(f"   ✓ card valid · auth={card['authSchemes'][0]['scheme']} · "
         f"skills={[s['id'] for s in card['skills']]}")

    step("2) MATCH — pick the skill for this request")
    skill_id = llm(f"Which skill handles: {user_msg}")
    skill = next((s for s in card["skills"] if s["id"] == skill_id), None)
    if not skill:
        return {"state": "failed", "reason": "no matching skill on card"}
    step(f"   ✓ matched skill '{skill_id}'")

    step("3) SUBMIT — send the task (validated against the skill's input_schema)")
    vendor = MockVendorAgent()
    inputs = {"sku": "COMP-24K-BTU", "room": "R305", "urgency": "rush"}
    task = vendor.submit(skill_id, inputs)
    step(f"   task {task.get('task_id')} → state={task['state']}")

    if task["state"] == "input-required":
        step(f"4) INPUT-REQUIRED — vendor asks: {task['question']!r}")
        step("   (this is HITL across a network boundary — surface to a human)")
        answer = "B"
        step(f"   caller answers: {answer!r} → resuming task")
        task = vendor.resume(task["task_id"], answer)

    step(f"5) {task['state'].upper()} — result: {task.get('result')}")
    return task


# ── The real shape (illustrative, not run) ──────────────────────────────────
def show_real_shape() -> None:
    snippet = '''    # pip install a2a-sdk   (or python_a2a; Google ADK wraps this as RemoteA2aAgent)
    #
    # VENDOR side — expose a card + accept tasks:
    #   class PartsAgent(A2AServer): def handle_message(self, msg): ...
    #   run_server(PartsAgent(), port=5001)   # card auto-served at /.well-known
    #
    # CALLER side — Google ADK composes a remote agent as if it were local:
    #   from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
    #   vendor = RemoteA2aAgent(
    #       name="acme_parts",
    #       agent_card="https://parts.acme-hvac.example.com/.well-known/agent-card.json",
    #   )
    #   coordinator = Agent(..., sub_agents=[vendor])  # delegation feels local'''
    step("Real A2A shape (a2a-sdk / ADK RemoteA2aAgent):")
    print(snippet)


def main():
    banner("CP5 · A2A — agent cards and the cross-framework handoff")

    step("The vendor's public Agent Card (/.well-known/agent.json):")
    print(json.dumps(VENDOR_CARD, indent=2))
    print()

    step("Coordinator delegates a part order over A2A:")
    task = orchestrator_delegate(
        VENDOR_CARD, "order the replacement compressor part from the vendor", MockLLM()
    )
    assert task["state"] == "completed"
    print()

    if have_a2a():
        step("an A2A SDK is installed — you could stand this up for real.")
    show_real_shape()
    print()
    step("KEY IDEA: you delegated to an agent you don't own, on a framework you")
    step("never imported — discovered via its card, paused on input-required,")
    step("resumed to completion. CP6 wires this into the durable fleet.")


if __name__ == "__main__":
    main()
