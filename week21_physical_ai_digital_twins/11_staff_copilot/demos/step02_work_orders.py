#!/usr/bin/env python3
"""PART 2 · Work-order intelligence — spatial context makes them good  [INTERMEDIATE]

A work order that just says "fix VAV-1-12" sends a technician to the wrong door with
the wrong part. The twin adds what the CMMS alone can't: asset → location → access
path → what else is nearby, so the visit is batched and the fix sticks. This demo
turns a natural-language complaint into that work order.

Run:  python demos/step02_work_orders.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402


def main() -> None:
    view.banner("PART 2", "Work-order intelligence — spatial context", "INTERMEDIATE")
    print("▣ MODE: SIM — the built-in twin stores answer every query (no endpoint, $0).")
    print("  a live twin serves the same calls from your graph DB / historian / CMMS API.\n")

    print("The outcome metrics every CMMS/twin vendor chases:")
    print("  • first-time-fix rate — fixed on the FIRST visit (right part, right access)")
    print("  • MTTR — mean time to repair, complaint → closed")
    print("  • truck rolls — site visits; each avoidable one is pure cost")
    print("  Vendors quote big % improvements on all three. Treat any specific number as")
    print("  marketing unless it comes with a source — none is quoted here.\n")

    complaint = "Guests in the west ballroom say it's warm and stuffy."
    print(f'COMPLAINT (natural language): "{complaint}"\n')

    print("→ STEP 1 · locate the asset on the twin (scene graph):")
    for s, p, o in (sim.graph_query(None, "serves", "West Ballroom")
                    + sim.graph_query("West Ballroom", "isPartOf")
                    + sim.graph_query("West Ballroom", "hasRoomRef")
                    + sim.graph_query("VAV-1-12", "locatedIn")
                    + sim.graph_query("VAV-1-12", "accessVia")):
        print(f"  ← ({s}) —{p}→ ({o})")
    print()

    print("→ STEP 2 · diagnose from graph + last-24 h telemetry:")
    damper = dict(sim.telemetry("VAV-1-12.damper_pos_pct"))
    temp = dict(sim.telemetry("West Ballroom.zone_temp_c"))
    sat = dict(sim.telemetry("AHU-2.supply_air_temp_c"))
    print(f"  ← damper: {damper[0]:.0f}% (h0) → {damper[23]:.0f}% flat since h6 "
          f"· zone temp: {temp[0]}→{temp[23]} °C · AHU-2 supply air ~{sat[12]} °C (normal)")
    hist = sim.cmms_history("VAV-1-12")[0]
    print(f"  ← CMMS: {hist[0]} · {hist[1]} · {hist[2]}")
    print("  ← diagnosis: damper actuator stuck — replace ACT-24M (repeat failure)\n")

    key, snip = sim.doc_search("vav damper actuator stuck")
    tech = sim.on_shift("hvac")[0]
    print("→ STEP 3 · draft the work order (everything a good WO needs):")
    print("  ┌───────────────────────────────────────────────────────────────────────")
    print("  │ WO-7411 · P2 · corrective                skill: HVAC/controls · est 45 min")
    print("  │ asset     VAV-1-12 (serves West Ballroom, fed by AHU-2)")
    print("  │ location  ceiling void above room B-102, Floor 1")
    print("  │ access    service corridor S-1 — BOH badge · ladder required")
    print("  │ diagnosis damper stuck at 12 % for 18 h; zone +3.7 °C; supply air normal")
    print("  │ parts     1× damper actuator ACT-24M · 1× linkage kit")
    print(f"  │ LOTO      {snip.split('LOCKOUT/TAGOUT: ')[1]}")
    print(f"  │ assigned  {tech['name']} ({tech['shift']})")
    print("  └───────────────────────────────────────────────────────────────────────\n")

    print("→ STEP 4 · batch the visit — what else is near that access path?")
    total = 0
    for pm, task, where, due, mins in sim.NEARBY_PM:
        total += mins
        print(f"  ← {pm} · {task} · {where} · {due} · +{mins} min")
    print(f"  ← one trip instead of three: +{total} min on site now, two truck rolls saved.\n")

    print("Takeaway: the twin's spatial context (location · access · nearby work) is what")
    print("moves first-time-fix, MTTR and truck rolls — the WO writes itself from the same")
    print("four stores Ch 2 queried. Next: WHO goes WHERE, in WHAT order — cuOpt dispatch.")


if __name__ == "__main__":
    main()
