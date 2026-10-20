#!/usr/bin/env python3
"""PART 3 · Cosmos world foundation models — generative worlds  [INTERMEDIATE]

Two engines, easy to confuse: OMNIVERSE is deterministic physics simulation (same
input → same answer, reproducible); COSMOS is a family of world foundation models
(WFMs) that GENERATE/PREDICT physically plausible worlds and video. Omniverse grounds;
Cosmos imagines — and together they feed synthetic data and policy training.

Run:  python demos/step03_cosmos_wfm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

DIAGRAM = """\
   OMNIVERSE (deterministic sim) ──physics-grounded scenes──► COSMOS (generative WFM)
        │  "what WILL happen"                                     │  "what COULD happen"
        │                                                         ▼
        │                                          synthetic data: rare events, variations
        ▼                                                         │
   validation · control decisions ◄── trained policies ◄── policy training (on the DGX)
"""


def main() -> None:
    view.banner("PART 3", "Cosmos world foundation models", "INTERMEDIATE")
    print("▣ MODE: offline — this chapter is pure stdlib (no GPU, no endpoint, $0).\n")

    print(DIAGRAM)
    print("Side by side — keep these two straight:\n")
    print(f"  {'dimension':<20}{'OMNIVERSE':<42}COSMOS")
    print("  " + "─" * 104)
    for dim, omni, cosmos in sim.OMNI_VS_COSMOS:
        print(f"  {dim:<20}{omni:<42}{cosmos}")
    print()

    print("How they work together (the NVIDIA pattern):")
    print("  • Omniverse supplies PHYSICS-GROUNDED simulation — the scene is right, the")
    print("    dynamics are right, every run is reproducible.")
    print("  • Cosmos supplies WFMs that generate/predict plausible variations of the")
    print("    world — the long tail your sensors will rarely record.")
    print("  • The NVIDIA Physical AI Data Factory Blueprint (GTC 2026) packages this")
    print("    pipeline for world modeling, humanoid skills and AV data at scale.\n")

    print("Where a BUILDING twin uses each:")
    print("  • Omniverse → the twin itself. 'What if we pre-cool floor 12 at 2pm?' needs")
    print("    a deterministic, reproducible answer you can act on — never a plausible guess.")
    print("  • Cosmos → synthetic data. Lobby crowding at check-in, smoke in a corridor,")
    print("    a decade of occupancy patterns you don't have — generate them to train the")
    print("    vision models (App 11) and control policies (App 09) safely.\n")

    print("Takeaway: Omniverse answers, Cosmos imagines — a twin needs the first to decide")
    print("and the second to train. Next: the Blueprints, and who actually runs this today.")


if __name__ == "__main__":
    main()
