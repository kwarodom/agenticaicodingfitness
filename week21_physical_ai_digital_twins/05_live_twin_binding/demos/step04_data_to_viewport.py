#!/usr/bin/env python3
"""PART 4 · From data to the viewport — heat maps, alerts, copilot  [ADVANCED]

The payoff chapter: live data becomes something an operator can SEE. Rooms get
heat-map colors from their live temperature deviation (displayColor opinions on
the session layer), a stuck-damper VAV gets an alert prim, and one short LLM
call translates Ch 4's natural-language question into a query plan across
graph + TSDB + USD. Everything authored here is a session-layer opinion —
delete the session layer and the building is pristine.

Run:  python demos/step04_data_to_viewport.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim   # noqa: E402
import view  # noqa: E402

PIPE = """\
   session layer / Fabric (Ch 3) ──► viewport RULES ──► what the operator sees
     iot:temperature per prim          Δ = live − setpoint → displayColor
     graph + TSDB (Ch 4)               fault detected      → alert prim
     NL question                       LLM                 → query plan → all three stores
"""


def main() -> None:
    view.banner("PART 4", "From data to the viewport", "ADVANCED")
    view.mode_line()

    print("Three ways the STATE layer becomes visible:\n")
    print(PIPE)

    # ── 1 · heat map: deviation from setpoint → session-layer displayColor ──────
    print("① HEAT MAP — floor 5, live temp vs setpoint → color (sim.heat_color):\n")
    print(f"  {'room':<10}{'live °C':>8}{'setpt':>7}{'Δ':>7}   color")
    print("  " + "─" * 56)
    for room, live, setpt, vav in sim.LIVE_ROOMS:
        delta = round(live - setpt, 1)
        name, rgb_ = sim.heat_color(delta)
        print(f"  {room:<10}{live:>8.1f}{setpt:>7.1f}{delta:>+7.1f}   {name}")
    print("\n  …authored as session-layer opinions (App 2's overrides, never edits):\n")
    for room, live, setpt, vav in sim.LIVE_ROOMS:
        _, rgb_ = sim.heat_color(live - setpt)
        prim = f"/Building/Floor5/Rooms/{room.replace('-', '_')}"
        col = "(" + ", ".join(f"{c:.2f}" for c in rgb_) + ")"
        print(f'  over "{prim}" {{ primvars:displayColor = [{col}] }}')
    print()

    # ── 2 · alert prim: the stuck damper Ch 4's TSDB series hinted at ───────────
    f = sim.DAMPER_FAULT
    p = next(p for p in sim.POINTS if p["tag"] == f["tag"] and p["kind"] == "damper position")
    print(f"② ALERT PRIM — {f['tag']}: damper commanded {f['cmd_pct']}%, actually at "
          f"{f['pos_pct']}% for {f['stuck_min']} min → STUCK.")
    print("   (That's why Room-503 ran hot every afternoon in Ch 4's series.)\n")
    print(f'  over "{p["prim"]}" {{')
    print(f'      def "Alert_stuck_damper" {{')
    print(f'          custom string alert:severity = "high"')
    print(f'          custom string alert:message  = "damper cmd {f["cmd_pct"]}% vs pos '
          f'{f["pos_pct"]}% for {f["stuck_min"]} min"')
    print("      }")
    print("  }   ← session layer again; the viewport renders a badge at the VAV\n")

    # ── 3 · the copilot: NL question → query plan over the three stores ─────────
    print("③ COPILOT — one short LLM call turns Ch 4's question into a query plan:\n")
    view.generate(
        "Translate this operator question into a query plan over our three twin "
        "stores (Brick semantics graph, TSDB, USD stage), joined on the IFC "
        "GlobalId: 'Show me all VAV boxes fed by AHU-3 on floor 5, with their "
        "last-24 h zone temps.'",
        max_tokens=350, title="Twin copilot — NL → query plan")
    print()

    print("Takeaway: the viewport is a RENDER of the state layer — colors, alerts and")
    print("answers are all session-layer opinions computed from graph + TSDB + live")
    print("attrs. The scene is now ALIVE; App 06 teaches it physics (EnergyPlus), and")
    print("Apps 10–12 put agents and copilots on top of exactly this query pattern.")


if __name__ == "__main__":
    main()
