#!/usr/bin/env python3
"""PART 1 · What is Physical AI? — the three-computer model  [BEGINNER]

Physical AI is AI that perceives, reasons and ACTS in the physical world — a robot,
a car, a BUILDING. NVIDIA's story runs on three computers: a DGX trains the models,
OVX/Omniverse simulates the world (sim2real), and Jetson/AGX deploys at the edge
where the building or robot actually is.

Run:  python demos/step01_three_computers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

DIAGRAM = """\
   ① DGX ── trained models/policies ──► ② OVX / Omniverse ── validated behavior ──► ③ Jetson AGX
      TRAIN                                 SIMULATE (sim2real)                        ACT (edge)
        ▲                                        │                                        │
        └──── experience & data flow back ◄──────┴───── telemetry from the real building ─┘
"""


def main() -> None:
    view.banner("PART 1", "Physical AI — the three-computer model", "BEGINNER")
    print("▣ MODE: offline — this chapter is pure stdlib (no GPU, no endpoint, $0).\n")

    print("Physical AI = AI that perceives, reasons and ACTS in the physical world.")
    print("A chatbot answers; Physical AI moves a robot arm, steers a car — or runs a")
    print("building's chillers. Acting in the real world is why the loop needs THREE computers:\n")
    print(DIAGRAM)

    print("The three computers, for a building twin:\n")
    for name, role, lives, runs, example in sim.THREE_COMPUTERS:
        print(f"  • {name} — {role}")
        print(f"      where it lives : {lives}")
        print(f"      what runs on it: {runs}")
        print(f"      week-21 example: {example}")
        print()

    print("The sim2real gap — and why simulation-first matters for buildings:")
    print("  • A policy trained in simulation behaves differently on the real thing —")
    print("    that difference is the sim2real gap. The better the simulation, the")
    print("    smaller the gap; closing it is what computer ② exists for.")
    print("  • For robots you can crash a few prototypes. For a building you can't:")
    print("    you can't A/B-test a chiller plant on hotel guests. The ONLY safe place")
    print("    to let an agent make mistakes is a twin of the building — that's why")
    print("    every later Week-21 app builds simulation-first, then deploys.\n")

    print("Takeaway: DGX trains, OVX simulates, AGX acts — and the twin is where an agent")
    print("earns the right to touch the real building. Next: what 'Omniverse' actually is in 2026.")


if __name__ == "__main__":
    main()
