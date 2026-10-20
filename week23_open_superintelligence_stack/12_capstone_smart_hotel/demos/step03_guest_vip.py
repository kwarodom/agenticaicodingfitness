#!/usr/bin/env python3
"""CH 4 · VIP comfort request — guardrails stop an unsafe autonomous action  [INTERMEDIATE]

A VIP in room 1512 asks for it cooler. The Guest specialist looks up the profile,
consults the SOP (RAG skill), and TRIES to change the setpoint — but the signed
OpenShell policy protects VIP-occupied rooms, so the action is denied and routed to
a human concierge. This is "what the agent may DO", enforced.

Run:  python demos/step03_guest_vip.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402
from hotel.runtime import build  # noqa: E402


def main() -> None:
    view.banner("CH 4", "VIP comfort request — policy-guarded", "INTERMEDIATE")
    view.mode_line()
    print("Guest request:  “VIP in room 1512 would like it a little cooler.”\n")

    rt = build()
    res = rt.orch.handle_event("VIP in room 1512 would like it cooler.", "1512")

    for a in res.actions:
        print(f"  → ACT {a['tool']}({a['args']})")
        print(f"  ← OBSERVE {a['result']}")
    for d in res.denials:
        print(f"  ⛔ POLICY DENIED — {d}")
    print(f"\n  · {res.answer}")
    print(f"  needs human approval: {res.needs_human}")
    print("\n".join(rt.relay.render_trace("VIP 1512 request")))
    print("\nTakeaway: comfort still wins for VIPs, but a human — not the agent — makes the")
    print("call. The deny reason is fed back so the agent adapts instead of forcing the action.")


if __name__ == "__main__":
    main()
