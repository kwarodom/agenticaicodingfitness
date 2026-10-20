#!/usr/bin/env python3
"""CH 3 · CRITICAL alarm, room 1203 — triage under guardrails  [INTERMEDIATE]

The room-1203 thread from the Week 23 tool-calling demo, now end-to-end: the Intent
Router sends the alarm to the Maintenance specialist (NemoClaw, on Nemotron Super),
which reads telemetry, checks the SOP via the NeMo Retriever RAG skill, and dispatches
a CRITICAL work order — every call gated by the signed OpenShell policy.

Run:  python demos/step02_critical_alarm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402
from hotel.runtime import build  # noqa: E402
from hotel.policy import signature_line  # noqa: E402


def main() -> None:
    view.banner("CH 3", "CRITICAL alarm, room 1203 — safe autonomous triage", "INTERMEDIATE")
    view.mode_line()
    print("Incoming BMS alarm:  “Room 1203 temperature alarm — triage and act.”")
    print(signature_line(), "\n")

    rt = build()
    res = rt.orch.handle_event("Room 1203 temperature alarm — triage and act.", "1203")

    for a in res.actions:
        print(f"  → ACT {a['tool']}({a['args']})")
        print(f"  ← OBSERVE {a['result']}")
    if res.denials:
        print(f"  ⚠ policy denied: {res.denials[0]}")
    print(f"\n  · {res.answer}")
    print(f"\n  work orders now open: {rt.state.work_orders}")
    print("\n".join(rt.relay.render_trace("room 1203 triage")))
    print("\nTakeaway: the Router right-sized this to Super (hard reasoning); the tool")
    print("normalizes 'Room 1203'→'1203'; the policy gateway let a safe CRITICAL dispatch through.")


if __name__ == "__main__":
    main()
