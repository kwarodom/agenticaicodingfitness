#!/usr/bin/env python3
"""PART 3 · The three-store architecture — geometry, meaning, history  [ADVANCED]

A production twin is never one database. USD holds GEOMETRY, a semantics GRAPH
holds MEANING (what feeds what, what is where), and a TSDB holds HISTORY — and
the IFC GlobalId / equipment tag joins all three. This demo answers the
canonical spatial-temporal question — "all VAV boxes fed by AHU-3 on floor 5,
with their last-24 h zone temps" — as one graph traversal + one TSDB fetch +
one USD highlight, then shows the BMS-spelunking alternative.

Run:  python demos/step03_three_stores.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim   # noqa: E402
import view  # noqa: E402

STORES = """\
  store             holds                          typical tech             its own key
  ────────────────  ─────────────────────────────  ───────────────────────  ────────────
  USD stage         GEOMETRY · prim hierarchy      Omniverse / usd-core     prim path
  semantics graph   MEANING · feeds/hasPart/…      Brick (RDF/OWL) · Neo4j  equipment tag
  TSDB              HISTORY · every reading ever   Timescale / Influx       point name
  ────────────────  ─────────────────────────────  ───────────────────────  ────────────
                    the join key across all three: the IFC GlobalId (App 3 preserved it)
"""

BLOCKS = "▁▂▃▄▅▆▇█"


def spark(series: list[float], lo: float, hi: float) -> str:
    span = max(hi - lo, 0.1)
    return "".join(BLOCKS[min(7, int((v - lo) / span * 7.999))] for v in series)


def main() -> None:
    view.banner("PART 3", "The three-store architecture", "ADVANCED")
    print("▣ MODE: SIM by design — the graph and TSDB run in-process (dict triples +")
    print("  a deterministic series), so the ARCHITECTURE is the lesson, not the infra.\n")

    print("One twin, three stores — each does what it's best at:\n")
    print(STORES)
    print("The standards for the MEANING store (don't invent your own ontology):")
    print("  • Brick Schema — RDF/OWL classes for equipment/points/spaces + the relationships")
    print("    feeds · hasPart · hasLocation · isPointOf (brickschema.org)")
    print("  • Project Haystack — point TAGGING conventions (project-haystack.org)")
    print("  • RealEstateCore — merged with Brick; Microsoft's DTDL-based REC ontology powers")
    print("    Azure Digital Twins (repo: WillowInc/opendigitaltwins-building)")
    print("  • ASHRAE Standard 223P — the standards-body semantics effort to watch\n")

    print("The comparison nobody spells out:")
    print("  • Azure Digital Twins = a semantic-GRAPH twin — rich meaning, NO geometry.")
    print("  • Omniverse           = a GEOMETRY twin — rich 3D, NO standard semantics.")
    print("  • A production building twin needs BOTH — hence three stores, one join key.")
    print("  Rule: USD is NOT the system of record for semantics — mirror onto prims only")
    print("  what the viewport needs; the graph owns the meaning.\n")

    # ── the canonical query, store by store ─────────────────────────────────────
    print('THE QUERY: "all VAV boxes fed by AHU-3 on floor 5, with last-24 h zone temps"\n')

    print("① GRAPH traversal (Brick relationships — Week 14/15's Neo4j/GraphRAG lives here):")
    hits = []
    for _, _, vav in sim.q("AHU-3", "feeds"):
        room = sim.q(vav, "hasLocation")[0][2]
        on_5 = bool(sim.q("Floor-5", "hasPart", room))
        mark = "✓" if on_5 else f"✗ excluded — {room} is not on Floor-5"
        print(f"   AHU-3 -feeds→ {vav:<9} -hasLocation→ {room:<9} {mark}")
        if on_5:
            hits.append((vav, room))
    print(f"   (VAV-3-07 never appears — AHU-2 feeds it) → {len(hits)} VAVs match\n")

    print("② TSDB fetch — follow isPointOf to each VAV's zone-temp point, last 24 h:")
    data = {}
    for vav, _ in hits:
        pt = next(s for s, _, _ in sim.q(None, "isPointOf", vav) if s.endswith(".ZNT"))
        data[vav] = sim.series_24h(pt)
    lo = min(min(s) for s in data.values())
    hi = max(max(s) for s in data.values())
    for vav, s in data.items():
        flag = "  ← runs HOT every afternoon (Ch 5 finds out why)" if max(s) > 25 else ""
        print(f"   {vav:<9} {spark(s, lo, hi)}  min {min(s):.1f} · max {max(s):.1f} °C{flag}")
    print()

    print("③ USD highlight — tag → GlobalId → prim path (the registry from Ch 2):")
    for vav, room in hits:
        p = next(p for p in sim.POINTS if p["tag"] == vav and p["kind"] == "zone air temp")
        print(f"   {vav:<9} GlobalId {p['guid']} → {p['prim']}   ({room})")
    print("   → author a highlight opinion per prim, on the SESSION layer only.\n")

    print("The alternative, without the three stores (BMS spelunking):")
    print("  • scroll a flat list of 3,000 point names hoping 'VAV5_03_RM_TEMP' is floor 5;")
    print("  • the AHU→VAV relationship exists only in a retired engineer's memory;")
    print("  • export CSVs per point, join by hand in a spreadsheet, repeat next week.")
    print("  One graph traversal + one range query + one prim map replaces ALL of it.\n")

    print("Takeaway: geometry, meaning and history live in different stores on purpose —")
    print("the GlobalId join makes them ONE twin. Next: paint this answer into the")
    print("viewport — heat maps, an alert prim, and an LLM planning the query for you.")


if __name__ == "__main__":
    main()
