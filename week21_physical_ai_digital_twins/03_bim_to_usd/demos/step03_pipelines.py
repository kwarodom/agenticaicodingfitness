#!/usr/bin/env python3
"""PART 3 · Pipelines in 2026 — the post-connector era  [INTERMEDIATE]

NVIDIA deprecated the Omniverse Launcher (EOL Oct 1 2025) and the first-party
connectors with it — so "install the Revit connector" is dead advice. This chapter
ranks the five real paths and simulates converting the SAME hotel wing through
three of them, printing a preservation scorecard.

Run:  python demos/step03_pipelines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim   # noqa: E402
import view  # noqa: E402

DIAGRAM = """\
   Revit / BIM ──► IFC4 file ──► converter ──► USD stage ──► (App 4 · assembly)
                      │              │
                      │              └─ what it KEEPS decides if the twin works
                      └─ the open, durable interchange — outlives any vendor app
"""


def main() -> None:
    view.banner("PART 3", "Pipelines in 2026 — the post-connector era", "INTERMEDIATE")
    print("▣ MODE: SIM — simulated conversions (stdlib only, no LLM, no GPU, $0).\n")

    print("Context: NVIDIA deprecated the Omniverse Launcher (EOL Oct 1 2025) and its")
    print("first-party connectors, in favor of native OpenUSD interchange. Don't learn a")
    print("connector — learn which PATH preserves what.\n")
    print(DIAGRAM)

    print("The five real paths, ranked for twin work:\n")
    for rank, path, verdict in sim.PIPELINES:
        print(f"  {rank}. {path}")
        print(f"     {verdict}")
    print()

    total = sum(1 for _ in sim.flatten())
    print(f"Simulated conversion — the same hotel wing ({total} IFC elements, 2 storeys)")
    print("pushed through paths 1, 4 and 5 (illustrative sim numbers):\n")
    print(f"  {'pipeline':<38}{'geometry':>9}{'Psets':>7}   GUIDs kept?")
    print("  " + "─" * 88)
    for label, geo, psets, guids in sim.SCORECARD:
        bar = "█" * max(1, round(psets / 100 * 20))
        print(f"  {label:<38}{geo:>8}%{psets:>6}%   {guids}")
        print(f"  {'':<38}{bar if psets else '·'}")
    print()

    print("Reading the scorecard:")
    print("  • Geometry survives everywhere — geometry was never the problem.")
    print("  • Psets and GUIDs are what separate a scene from a twin: path 1 keeps them as")
    print("    prim attrs, path 4 buries them in Datasmith metadata, path 5 deletes them.")
    print("  • Path 5 (FBX/glTF) LOOKS fine in a viewport — and is unusable for a twin.\n")

    print("run it for real (path 1, the open one):")
    print("  $ pip install ifcopenshell        # ifcopenshell.org — open-source IFC toolkit")
    print("  # then: load the IFC, walk the spatial tree, write prims + ifc:* attrs — Ch 5")
    print("  # builds exactly that converter, live.\n")

    print("Takeaway: pick the pipeline by what SURVIVES, not by what renders. IFC-first")
    print("(path 1) is the most metadata-faithful open path. Next: build the converter.")


if __name__ == "__main__":
    main()
