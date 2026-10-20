#!/usr/bin/env python3
"""PART 1 · Define a specialized agent  [BEGINNER]

A NemoClaw agent is a SPEC, not hand-written code: a role/persona, a base open
model (Nemotron Nano/Super), a system prompt, allowed skills & tools, and a signed
policy. This demo builds one such spec and prints it, then (in REAL mode) asks the
freshly-defined specialist to introduce itself so you can hear the persona.

Run:  python demos/step01_define_agent.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import view  # noqa: E402

# The NemoClaw agent spec — author once, run as a domain expert.
AGENT_SPEC = {
    "name": "hvac-reliability-expert",
    "role": "HVAC reliability engineer for a hotel portfolio",
    "base_model": "nemotron-3-nano:30b-a3b",   # Nano = cheap, many specialists
    "system_prompt": ("You are an HVAC reliability engineer. Diagnose comfort + energy "
                      "issues from telemetry, cite the reading, and act only within policy."),
    "skills": ["nemo-retriever (runbook RAG)", "cuopt (setpoint optimization)"],
    "tools": ["get_room_telemetry", "dispatch_work_order"],
    "policy": {"sandbox": "openshell", "egress_allowlist": ["telemetry.internal"],
               "signed": True, "may_write": False},
}


def main() -> None:
    view.banner("PART 1", "Define a specialized agent", "BEGINNER")
    view.mode_line()

    print("A NemoClaw agent = base model + persona + skills + tools + signed policy.")
    print("This is the authored SPEC (not code) — the harness around the model:\n")
    print(json.dumps(AGENT_SPEC, indent=2))
    print()

    print("Read it top-to-bottom:")
    print(f"  • role     → {AGENT_SPEC['role']}")
    print(f"  • model    → {AGENT_SPEC['base_model']} (the open Nemotron it's built on)")
    print(f"  • skills   → {', '.join(AGENT_SPEC['skills'])}")
    print(f"  • tools    → {', '.join(AGENT_SPEC['tools'])}")
    print(f"  • policy   → sandbox={AGENT_SPEC['policy']['sandbox']}, "
          f"signed={AGENT_SPEC['policy']['signed']}, "
          f"egress={AGENT_SPEC['policy']['egress_allowlist']}\n")

    print("Meet the specialist (its persona in one line):\n")
    view.generate("You are the HVAC reliability engineer defined above. In one sentence, "
                  "state your role and what you are and are not allowed to do.",
                  max_tokens=200, title="the specialist introduces itself")

    print("\nTakeaway: the spec IS the specialist. Next: equip it with real skills & tools.")


if __name__ == "__main__":
    main()
