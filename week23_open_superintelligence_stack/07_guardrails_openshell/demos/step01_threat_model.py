#!/usr/bin/env python3
"""PART 1 · Why guard a long-running sovereign agent?  [BEGINNER]

Agent = Model + Harness. Weeks 1-5 gave the harness POWER — tools, long-running
state, egress. Power without rails is a liability. This chapter lays out the threat
model for a sovereign agent living for days on your DGX, and the three-layer defense.

Run:  python demos/step01_threat_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402

THREATS = [
    ("Prompt injection",   "user/doc text hijacks the agent's instructions",
     "NeMo input rail — jailbreak detector"),
    ("Data exfiltration",  "a tool call ships secrets to an attacker host",
     "OpenShell egress allowlist"),
    ("Jailbreak",          "'ignore all rules' → unsafe output",
     "NeMo input + output rails"),
    ("Unbounded egress",   "agent phones home / reaches the open internet",
     "OpenShell allowlist + sandbox"),
    ("PII leakage",        "sensitive data routed to a cloud model",
     "Privacy router → LOCAL NIM"),
]

LAYERS = """\
  ┌─────────────────────────────────────────────────────────────┐
  │  NeMo Guardrails   — controls WHAT THE MODEL SAYS             │
  │     input rails · topic/dialog rails · output rails          │
  ├─────────────────────────────────────────────────────────────┤
  │  OpenShell runtime — controls WHAT THE TOOLS CAN DO           │
  │     sandbox · network egress allowlist · signed policy       │
  ├─────────────────────────────────────────────────────────────┤
  │  Privacy router    — controls WHERE THE DATA GOES            │
  │     PII/secrets → LOCAL sovereign NIM, never the perimeter   │
  └─────────────────────────────────────────────────────────────┘"""


def main() -> None:
    view.banner("PART 1", "Why guard a long-running sovereign agent?", "BEGINNER")
    view.mode_line()

    print("A sovereign agent runs for DAYS on your DGX with tools, memory, and network.")
    print("That is exactly what an attacker wants. The threat model:\n")

    print(f"  {'Threat':<20}{'What goes wrong':<48}Defense")
    print("  " + "─" * 96)
    for t, what, defense in THREATS:
        print(f"  {t:<20}{what:<48}{defense}")
    print()

    print("The layered defense (defense in depth — one layer failing is not a breach):\n")
    print(LAYERS)
    print()
    print("Key idea: Guardrails is NOT the same as OpenShell. Rails police the model's")
    print("words; the secure runtime polices the tools' actions; the router polices data")
    print("residency. You need all three for a long-running sovereign agent.\n")

    print("Takeaway: securing an agent is three jobs — what it SAYS, what it DOES, and")
    print("where its DATA GOES. The next chapters build each layer.")


if __name__ == "__main__":
    main()
