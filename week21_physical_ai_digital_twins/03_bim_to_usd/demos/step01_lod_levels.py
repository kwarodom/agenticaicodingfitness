#!/usr/bin/env python3
"""PART 1 · BIM LOD, properly  [BEGINNER]

LOD = Level of DEVELOPMENT — how much you can RELY on an element for decisions —
not level of detail. The BIMForum LOD Specification (building on AIA G202-2013)
defines it per element: Part I is geometry, Part II is attribute tables (~COBie-
aligned). This chapter drills the twin-readiness bar: which LOD a twin needs.

Run:  python demos/step01_lod_levels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim   # noqa: E402
import view  # noqa: E402


def main() -> None:
    view.banner("PART 1", "BIM LOD, properly", "BEGINNER")
    print("▣ MODE: SIM — offline table walkthrough (stdlib only, no LLM, no GPU, $0).\n")

    print("LOD = Level of DEVELOPMENT, not detail. It measures how much you can RELY on")
    print("an element for decisions. The trap: a detailed manufacturer mesh dropped into a")
    print("concept-stage model is still LOD 200 — pretty geometry does not add reliability.\n")

    print("The BIMForum LOD Specification (builds on AIA G202-2013) defines, per element:")
    print("  • Part I  — the geometry you may rely on at each level")
    print("  • Part II — attribute tables (roughly COBie-aligned) — the DATA side\n")

    print(f"  {'LOD':<5}{'name':<26}what the geometry means / twin relevance")
    print("  " + "─" * 86)
    for lod, name, geo, twin in sim.LOD_TABLE:
        print(f"  {lod:<5}{name:<26}{geo}")
        print(f"  {'':<31}twin: {twin}")
    print()

    print("Two levels people get wrong:")
    print("  • LOD 350 = LOD 300 + cross-trade INTERFACES (supports, hangers, clearances) —")
    print("    the clash-coordination level, and the right bar for maintained MEP.")
    print("  • LOD 500 = FIELD-VERIFIED as-built — an attestation, NOT more geometry.")
    print("    BIMForum deliberately defines no LOD 500 geometry.\n")

    print("Europe note: ISO 19650 / EN 17412-1 replaced LOD with the Level of Information")
    print("Need (LOIN) — same idea, stated as 'what information does this decision need?'.\n")

    print("The twin-readiness bar (drill this):")
    print("  ✓ geometry at LOD 200–300 (350 for maintained MEP) is ENOUGH for a twin")
    print("  ✗ LOD 400 fabrication meshes actively HURT a real-time twin — polygon bloat")
    print("    that the render loop pays for on every frame, for zero operational value")
    print("  ✓ what actually matters is LOD-500-grade DATA — the COBie handover:")
    print("    manufacturer · model · serial · install date · warranty · zone\n")

    print('Takeaway: "A twin fails on missing data far more often than missing geometry."')
    print("Ask for LOD 300 geometry and LOD-500-grade data — not the prettiest mesh.")
    print("Next: what actually survives the trip from BIM into USD.")


if __name__ == "__main__":
    main()
