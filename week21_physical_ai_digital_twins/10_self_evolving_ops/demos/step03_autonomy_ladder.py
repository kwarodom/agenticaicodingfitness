#!/usr/bin/env python3
"""PART 3 · The safe-autonomy ladder  [ADVANCED]

A learning policy earns the right to touch a real BMS one rung at a time: OFFLINE →
SHADOW → ADVISORY → SUPERVISED AUTONOMY, each rung with an explicit, numeric gate.
The key principle: constrain the ACTION SPACE, not just the reward — clamps and
masking beat penalty terms for safety.

Run:  python demos/step03_autonomy_ladder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

LADDER_ART = """\
   rung ④  SUPERVISED AUTONOMY   agent WRITES setpoints — inside hard guardrails
   rung ③  ADVISORY              agent RECOMMENDS — a human approves every write
   rung ②  SHADOW                agent COMPUTES — nothing written, divergence logged
   rung ①  OFFLINE               agent TRAINS — simulation only (Sinergym / BOPTEST)
"""


def main() -> None:
    view.banner("PART 3", "The safe-autonomy ladder", "ADVANCED")
    print("▣ MODE: offline — this chapter is pure stdlib (no GPU, no endpoint, $0).\n")

    print("You can't A/B-test a chiller plant on hotel guests. Autonomy is granted one")
    print("rung at a time, and each promotion has a gate you can put a number on:\n")
    print(LADDER_ART)

    print("Walking ONE policy up the ladder — the gate criteria, printed per rung:\n")
    for rung, metric, result, gate in sim.LADDER_WALK:
        print(f"  {rung}")
        print(f"     measure : {metric}")
        print(f"     result  : {result}")
        print(f"     {gate}")
        print()

    print("Rung ④'s HARD guardrails (these never come off):\n")
    for name, what in sim.GUARDRAILS:
        print(f"  • {name:<27} {what}")

    print("\nThe key principle — constrain the ACTION SPACE, not just the reward:")
    print("  • A reward penalty says 'writing 18 °C costs you points' — a trained policy")
    print("    can still emit it on an out-of-distribution day, and the building pays.")
    print("  • A clamp / action mask says 'writes outside 22–27 °C do not exist' — the")
    print("    unsafe action is unrepresentable, whatever the model does.")
    print("  • Penalties shape behavior; clamps bound it. Safety cases are built on bounds.")

    print("\nReal-world calibration — who runs learned control on buildings today:\n")
    for who, what in sim.CALIBRATION:
        print(f"  • {who}")
        print(f"      {what}")
    print(f"\n  Honesty note: {sim.CALIBRATION_NOTE}")

    print("\n🖥️  run it for real — rung ① offline gyms on your own machine/DGX:")
    print("   $ pip install sinergym            # EnergyPlus HVAC gym (App 9)")
    print("   $ docker run -p 80:80 boptest     # BOPTEST: Modelica benchmark, fixed KPIs")

    print("\nTakeaway: shadow before advisory, advisory before autonomy, guardrails forever.")
    print("Next: live one guardrailed week — a sensor will fail, and the ladder will hold.")


if __name__ == "__main__":
    main()
