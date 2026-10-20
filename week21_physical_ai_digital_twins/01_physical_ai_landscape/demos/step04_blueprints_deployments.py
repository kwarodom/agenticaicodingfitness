#!/usr/bin/env python3
"""PART 4 · Blueprints & real deployments — honestly tiered  [ADVANCED]

NVIDIA ships digital-twin BLUEPRINTS (reference architectures): Mega for industrial/
warehouse robot fleets, DSX for gigawatt-scale AI factories. This demo lists what is
PROVEN vs FRONTIER — and asks the model to apply the DSX 'twin as operating system'
idea to a Bangkok hotel (the one honest LLM call in this app).

Run:  python demos/step04_blueprints_deployments.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

LIFECYCLE = """\
   DSX lifecycle:  DESIGN ──► SIMULATE ──► BUILD ──► the twin becomes the facility's
                   (twin first, concrete later)       OPERATING SYSTEM: monitor · inspect · optimize
"""


def main() -> None:
    view.banner("PART 4", "Blueprints & real deployments — honestly tiered", "ADVANCED")
    view.mode_line()

    print("The two flagship Omniverse Blueprints:")
    print("  • Mega — industrial/warehouse twins to TRAIN + VALIDATE physical-AI robot")
    print("    fleets before they touch the real floor (the Amazon Robotics pattern).")
    print("  • DSX (GTC 2026, GA with the Vera Rubin AI Factory reference design) —")
    print("    gigawatt-scale AI-factory digital twins. Design → simulate → then, once")
    print("    live, the twin becomes the facility's 'operating system'.\n")
    print(LIFECYCLE)

    print("Who actually runs Omniverse twins today (tiered honestly):\n")
    print(f"  {'deployment':<30}{'domain':<14}{'tier':<11}what happened")
    print("  " + "─" * 108)
    for name, domain, tier, what in sim.DEPLOYMENTS:
        mark = "✓" if tier == "PROVEN" else "⚠"
        print(f"  {name:<30}{domain:<14}{mark} {tier:<9}{what}")
    print()

    print("The honesty note (read it twice):")
    print("  • Proven wins are factories, warehouses, retail and data centers. Hotels,")
    print("    hospitals and offices on Omniverse are FRONTIER — no established reference")
    print("    deployments. That is not a reason to skip Week 21; it is WHY Week 21")
    print("    teaches durable format-level skills (USD, IFC, Brick, FMI) instead of")
    print("    betting your career on one vendor app.\n")

    view.generate(
        "Apply the DSX Blueprint's 'twin as the facility's operating system' idea to a "
        "30-floor Bangkok hotel. Name the 4 twin layers from the Week 21 stack (SCENE, "
        "STATE, SIMULATION, AGENTS) and say in one clause what each does for the hotel. "
        "2-3 sentences total.",
        max_tokens=300, title="DSX for a hotel — twin as operating system")

    print("\nTakeaway: the Blueprints prove the pattern (twin first, then the twin runs the")
    print("facility); buildings are the frontier we build toward. Next: App 02 — OpenUSD,")
    print("the format the whole SCENE layer stands on.")


if __name__ == "__main__":
    main()
