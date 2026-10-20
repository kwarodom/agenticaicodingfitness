#!/usr/bin/env python3
"""PART 1 · Why surrogates — the latency ladder  [BEGINNER]

A twin viewport needs answers in MILLISECONDS while a facilities manager drags a
slider — but a full CFD run takes minutes to hours, and even an annual EnergyPlus
run takes minutes. The pattern that closes that gap: SOLVER for truth, SURROGATE
for interaction. The proof it works is shipping today in data-center twins.

Run:  python demos/step01_why_surrogates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402

LADDER = [
    ("CFD (airflow/thermal field)", "minutes–hours", "the TRUTH for air movement — design "
     "studies, hot-spot forensics, 'is this rack/ballroom layout safe?'"),
    ("EnergyPlus (annual energy)",  "minutes",       "the truth for yearly kWh — retrofits, "
     "tariffs, compliance (App 6)"),
    ("RC reduced-order model",      "seconds→ms",    "fast physics for MPC and hourly "
     "what-if — credible only after the App 7 calibration gate"),
    ("trained ML surrogate",        "milliseconds",  "INTERACTION — a slider drag, a "
     "viewport frame, a thousand-scenario sweep"),
]

DIAGRAM = """\
   the slider problem:
     manager drags "setpoint 24→25.5 °C" ──► twin viewport must redraw the
     temperature field NOW (≤ ~100 ms) ──► no solver on the ladder's top
     rungs can answer that fast ──► so the twin serves a SURROGATE trained
     on solver runs, and keeps the solver for truth.
"""


def main() -> None:
    view.banner("PART 1", "Why surrogates — the latency ladder", "BEGINNER")
    print("▣ MODE: REAL — pure-stdlib arithmetic; nothing to install, no endpoint needed.\n")

    print(DIAGRAM)
    print("The latency ladder — what each tool costs and what it's good for:\n")
    print(f"  {'tool':<30}{'one answer':>14}   good for")
    print("  " + "─" * 96)
    for tool, latency, good in LADDER:
        print(f"  {tool:<30}{latency:>14}   {good}")
    print()

    print("The canonical proof — data-center digital twins (shipping products, not demos):")
    print("  • Cadence Reality DC — a real product built on Omniverse APIs: CFD surrogates")
    print("    give interactive thermal what-if for rack placement and cooling failures.")
    print("  • The Omniverse Blueprint for AI-factory design & operations (GTC 2025) wires")
    print("    Schneider Electric (power/cooling), ETAP (electrical sim) and Vertiv into")
    print("    one twin — commonly quoted speedups for the surrogate path: 100–1000×.")
    print("  Why data centers first? One thermal question — 'will this rack overheat?' —")
    print("  asked thousands of times a day. Exactly the shape a surrogate serves best.\n")

    print("The pattern, plainly:")
    print("  • SOLVER for truth — run CFD/EnergyPlus offline, across many parametric cases.")
    print("  • SURROGATE for interaction — train ML on those runs; serve it in the viewport.")
    print("  • The solver never goes away: it generates training data, validates the")
    print("    surrogate, and answers anything outside what the surrogate has seen.\n")

    print("For a hotel: same ladder, different rooms — ballroom airflow, kitchen exhaust,")
    print("the chiller-plant room. The manager's slider doesn't care which building it is.\n")

    print("Takeaway: milliseconds is a hard requirement of interaction, and no solver meets")
    print("it — so twins pair SOLVER (truth) with SURROGATE (speed). Next: the NVIDIA")
    print("framework built for exactly this — PhysicsNeMo.")


if __name__ == "__main__":
    main()
