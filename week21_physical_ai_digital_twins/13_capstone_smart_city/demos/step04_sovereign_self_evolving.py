#!/usr/bin/env python3
"""CH 5 · Sovereign & self-evolving — the fleet that runs the city  [ADVANCED]

Two audits before any agent touches a city. SOVEREIGNTY: what data exists, where
it may flow, what never leaves the city's own GPUs (the blueprint's local /
local_shared / remote-NIM modes made concrete — Week 18's lesson with a city's
stakes). AUTONOMY: rungs are earned PER ACTION TYPE — publishing a verified alert
is rung 4, substation switching is rung 1 forever. Then the flywheel: tonight's
incidents become tomorrow's playbooks.

Run:  python demos/step04_sovereign_self_evolving.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402
from city import agents  # noqa: E402

RUNG = {1: "1 OFFLINE", 2: "2 SHADOW", 3: "3 ADVISORY", 4: "4 SUPERVISED"}


def main() -> None:
    view.banner("CH 5", "Sovereign & self-evolving — the city fleet", "ADVANCED")

    print("SOVEREIGNTY AUDIT — every data class, classified (Week 18, city stakes):\n")
    print(f"  {'data class':<30}{'class':<11}rule")
    print("  " + "─" * 76)
    for name, cls, rule in agents.DATA_CLASSES:
        print(f"  {name:<30}{cls:<11}{rule}")
    print()

    print("Model placement — the blueprint's three modes, honestly labeled:\n")
    for mode, what, verdict in agents.MODEL_MODES:
        print(f"  {mode:<14}{what:<42}{verdict}")
    print()

    print("AUTONOMY — rungs earned per ACTION TYPE (a city is not a hotel):\n")
    print(f"  {'action':<38}{'rung':<14}bound")
    print("  " + "─" * 78)
    for action, rung, bound in agents.ACTION_LADDER:
        print(f"  {action:<38}{RUNG[rung]:<14}{bound}")
    print()
    print("  Hard guardrails (the action space is constrained, not just the reward):")
    for g in agents.GUARDRAILS:
        print(f"    • {g}")
    print()

    print("THE FLYWHEEL — tonight's episodes become durable memory:\n")
    fw = agents.Flywheel()
    print(f"  before: {len(fw.episodic)} episodes · {len(fw.semantic)} facts · "
          f"{len(fw.procedural)} playbooks")
    fw.record({"kind": "incident", "where": "I-11", "route": "S-01/S-02",
               "reroute_worked": True, "vc_after": 0.89})
    fw.record({"kind": "grid", "sub": "SUB-B", "shortfall_mw": 0})
    new = fw.consolidate()
    print("  consolidation distils:")
    for item in new:
        print(f"    + {item}")
    print(f"  after:  {len(fw.episodic)} episodes · {len(fw.semantic)} facts · "
          f"{len(fw.procedural)} playbooks\n")

    print("The postmortem note the consolidator files (LLM-written — REAL if connected):")
    view.generate(
        "Write a 4-sentence postmortem for consolidation into semantic memory: a "
        "sideswipe at Sukhumvit x Asoke 23:15 was CV-detected in 2 s, VLM-verified in "
        "9 s, corridor reroute approved by the duty engineer in 3 min (advisory rung), "
        "diversion held v/c at 0.89. Recommend what to store and whether signal retiming "
        "should stay at advisory.",
        max_tokens=220, title="consolidator · postmortem → semantic memory")
    print()
    print("Takeaway: sovereign means the video never leaves your GPUs; self-evolving")
    print("means the city gets easier to run every night; safe means autonomy is earned")
    print("per action, with substation switching at rung 1 forever. Next: one full night.")


if __name__ == "__main__":
    main()
