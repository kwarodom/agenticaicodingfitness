#!/usr/bin/env python3
"""CH 6 · Self-improve — the Data Flywheel with verifiable rewards  [ADVANCED]

Runs a morning sweep, then scores each decision with a VERIFIABLE reward (NeMo Gym
style — an objective check, not a human vibe): did maintenance dispatch a genuinely
critical room? did energy actually save kW within comfort? was the VIP protected?
Clean, high-reward traces are curated to distill a cheaper Nano student later.

Run:  python demos/step05_flywheel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402
from hotel.runtime import build  # noqa: E402


def main() -> None:
    view.banner("CH 6", "Self-improve — Data Flywheel + verifiable rewards", "ADVANCED")
    view.mode_line()

    rt = build()
    brief = rt.orch.morning_brief()
    for role, rid, res in brief["results"]:
        rt.fly.observe(res, rt.state.rooms.get(rid))
        print(f"  logged [{role}] room {rid}: {res.answer[:64]}")
    print()
    print("\n".join(rt.fly.report(rt.relay)))
    print("\nTakeaway: the agent improves from its OWN traffic. Verifiable rewards make")
    print("'better' measurable; the flywheel turns yesterday's operations into a cheaper,")
    print("equally-good model tomorrow — self-evolving, on-box, sovereign.")


if __name__ == "__main__":
    main()
