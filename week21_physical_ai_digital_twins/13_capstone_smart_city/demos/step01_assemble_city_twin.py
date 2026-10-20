#!/usr/bin/env python3
"""CH 2 · Assemble the city twin — a twin of twins  [BEGINNER]

The SCENE rules from Apps 2–4, at city scale: districts are PAYLOADS (load one,
not four), 2,900 streetlights are 3 instanced prototypes, and whole buildings
compose in as REFERENCES to their own stages — Capstone I's Grand Bangkok hotel
twin is referenced here unchanged. Then the readiness validator gates go-live.

Run:  python demos/step01_assemble_city_twin.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402
from city import scene, world  # noqa: E402


def main() -> None:
    view.banner("CH 2", "Assemble the city twin — a twin of twins", "BEGINNER")

    n = world.counts()
    print("Krung Alto — the inventory (every entity carries an IFC-style GlobalId):\n")
    print(f"  {n['districts']} districts · {n['segments']} road segments · "
          f"{n['intersections']} intersections · {n['cameras']} cameras · "
          f"{n['substations']} substations · {n['buildings']} grid-interactive buildings\n")

    print("Layer stack (discipline federation, App 2 Ch 3 — city edition):\n")
    for i, layer in enumerate(scene.LAYERS):
        print(f"  {'└─' if i == len(scene.LAYERS)-1 else '├─'} {layer}")
    print()

    print("The composed stage (working set: Sukhumvit district only):\n")
    for line in scene.tree():
        print(f"  {line}")
    print()

    print("Instancing at city scale (the App 2 Ch 4 lesson, bigger numbers):\n")
    print(f"  {'asset':<18}{'count':>8}{'prototypes':>12}")
    print("  " + "─" * 40)
    for name, count, protos in scene.INSTANCED:
        print(f"  {name:<18}{count:>8}{protos:>12}")
    print()

    print("The nesting move — a twin of twins:")
    for r in scene.compose()["referenced_twins"]:
        print(f"  {r['path']}")
        print(f"    reference → {r['reference']}")
    print("  The hotel keeps its own layers, telemetry and validator; the city stage")
    print("  composes it without copying it. Fix the hotel once, the city sees it.\n")

    print("Twin-readiness validation (same gate as Capstone I):\n")
    for name, verdict, note in scene.validate():
        mark = "✓" if verdict == "PASS" else "⚠"
        print(f"  [{verdict:4}] {mark} {name:<48} {note}")
    print()
    print("Takeaway: a city twin is composition all the way down — payloads for")
    print("districts, references for building twins, instancing for the street. The")
    print("two WARNs are (again) missing DATA, not geometry. Next: give it eyes.")


if __name__ == "__main__":
    main()
