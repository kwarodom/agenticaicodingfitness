#!/usr/bin/env python3
"""CH 2 · Morning ops brief — the AI-Q Deep Agent runs the building  [BEGINNER]

The Intent Router escalates to the Deep Agent (Nemotron Super), which plans a
morning sweep and fans work out to the NemoClaw specialists (Energy, Maintenance,
Guest). Every action passes the OpenShell policy and is observed by NeMo Relay.

Run:  python demos/step01_morning_brief.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402
from hotel.runtime import build  # noqa: E402


def main() -> None:
    view.banner("CH 2", "Morning ops brief — AI-Q Deep Agent + specialists", "BEGINNER")
    view.mode_line()

    rt = build()
    print(f"▣ {rt.state.name} — {rt.state.now}")
    s = rt.state.summary()
    print(f"  occupancy {s['occupancy']} · energy {s['energy_kw']}kW "
          f"(target {s['target_kw']}kW, {s['over_target_kw']}kW over) · "
          f"tickets {s['tickets_open']} ({s['tickets_critical']} critical)\n")

    brief = rt.orch.morning_brief()
    for role, rid, res in brief["results"]:
        print(f"── {role.upper()} · room {rid}")
        for a in res.actions:
            print(f"  → ACT {a['tool']}({a['args']})")
            print(f"  ← OBSERVE {a['result']}")
        if res.denials:
            print(f"  ⚠ policy: {res.denials[0]}")
        print(f"  · {res.answer}\n")

    print(brief["policy"])
    print("\n".join(rt.relay.render_trace("morning brief")))
    print("\nTakeaway: one orchestrator, three specialized sub-agents, every action")
    print("policy-gated and traced — a real autonomous ops loop, running on your DGX.")


if __name__ == "__main__":
    main()
