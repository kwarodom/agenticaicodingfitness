#!/usr/bin/env python3
"""PART 1 · Copilot over the twin — four tools, one grounded answer  [BEGINNER]

A staff copilot is an LLM agent whose TOOLS are the twin's four stores: the semantics
graph (Brick/RealEstateCore/ASHRAE 223P), the time-series historian, the CMMS, and an
O&M-manual document RAG. One duty-manager question walks through all four — every tool
call printed — and ends in an answer grounded in the building, not in vibes.

Run:  python demos/step01_copilot_tools.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim   # noqa: E402
import view  # noqa: E402

DIAGRAM = """\
   staff question ─► [COPILOT · LLM agent] ─► ① graph.query      (semantics: Brick/REC/223P)
                            │                 ② tsdb.query       (last-24 h telemetry)
                            │                 ③ cmms.history     (work orders · who's on shift)
                            │                 ④ docs.search      (O&M-manual RAG)
                            └────────────────► grounded answer  (every claim traceable to a store)
"""


def main() -> None:
    view.banner("PART 1", "Copilot over the twin — four tools, one grounded answer", "BEGINNER")
    view.mode_line()

    print("The same twin the agent fleet controls (App 10) also makes the HUMANS faster.")
    print("Vendors ship assistants in exactly this shape — Willow Copilot, Johnson Controls")
    print("OpenBlue, Siemens Building X assistants (names to know; no feature claims here).")
    print("Under all of them is one pattern: an LLM agent with the twin's stores as tools.\n")
    print(DIAGRAM)

    print("Warm-up — why the graph tool matters (the Week 15 GraphRAG pattern, in a building):")
    print('  » graph.query(s="AHU-3", p="feeds") ∩ isPartOf "Floor 5"')
    fed = {o for _, _, o in sim.graph_query("AHU-3", "feeds")}
    on5 = sorted(v for v in fed if sim.graph_query(v, "isPartOf", "Floor 5"))
    print(f"  ← {', '.join(on5)}   ({len(on5)} VAVs — one query, not a BMS spelunking session)\n")

    q = "Why is the west ballroom warm and who should fix it?"
    print(f'DUTY MANAGER asks: "{q}"\n')

    print('→ TOOL 1/4 · graph.query("West Ballroom") — what serves it, where is it?')
    for s, p, o in (sim.graph_query(None, "serves", "West Ballroom")
                    + sim.graph_query(None, "feeds", "VAV-1-12")
                    + sim.graph_query("West Ballroom") + sim.graph_query("VAV-1-12")[1:]):
        print(f"  ← ({s}) —{p}→ ({o})")
    print()

    print("→ TOOL 2/4 · tsdb.query — last 24 h on the suspect points:")
    pts = ["West Ballroom.zone_temp_c", "VAV-1-12.damper_pos_pct", "AHU-2.supply_air_temp_c"]
    hours = [0, 4, 6, 8, 12, 18, 23]
    print(f"  {'point':<30}" + "".join(f"{f'h{h}':>7}" for h in hours))
    for p in pts:
        series = dict(sim.telemetry(p))
        print(f"  {p:<30}" + "".join(f"{series[h]:>7}" for h in hours))
    print("  ← damper flat at 12 % since ~h6 while zone temp climbs; supply air normal →")
    print("    the fault is at the terminal unit, not the AHU.\n")

    print('→ TOOL 3/4 · cmms.history("VAV-1-12") + who is on shift:')
    for wo, date, what in sim.cmms_history("VAV-1-12"):
        print(f"  ← {wo} · {date} · {what}")
    t = sim.on_shift("hvac")[0]
    print(f"  ← on shift with 'hvac': {t['name']} ({', '.join(t['skills'])} · {t['shift']})\n")

    print('→ TOOL 4/4 · docs.search("VAV damper actuator stuck"):')
    key, snip = sim.doc_search("VAV damper actuator stuck")
    print(f"  ← [{key}] {snip}\n")

    view.generate(
        "You are a building staff copilot. Tool results: VAV-1-12 serves West Ballroom "
        "(room B-102, floor 1), fed by AHU-2; damper stuck at 12% for 18h while zone temp "
        "rose 23.1→26.8C and AHU-2 supply air stayed ~13C; CMMS shows the ACT-24M actuator "
        "was replaced 14 months ago for sticking; O&M §4.2 says replace the actuator and "
        "lock out the 24V circuit at CP-1-B; Somchai (hvac/controls) is on shift. Answer "
        "the duty manager in 4 sentences: why is the ballroom warm and who should fix it?",
        max_tokens=320, title="Copilot synthesis — the grounded answer")

    print("\nTakeaway: four cheap tool calls beat one confident hallucination — every claim in")
    print("the answer traces to a store. Next: turning that diagnosis into a GOOD work order.")


if __name__ == "__main__":
    main()
