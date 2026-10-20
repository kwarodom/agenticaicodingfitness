#!/usr/bin/env python3
"""PART 2 · What survives BIM → USD  [INTERMEDIATE]

The IFC spatial tree maps beautifully onto a USD prim hierarchy — but USD is scene
description, not BIM authoring, so the parametric intelligence never survives.
This chapter shows the map, the ALWAYS / USUALLY / NEVER survival table, and THE
load-bearing rule: preserve the IFC GlobalId on every prim.

Run:  python demos/step02_what_survives.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim   # noqa: E402
import view  # noqa: E402

TREE_MAP = """\
   IFC spatial tree                          USD prim hierarchy
   ────────────────                          ──────────────────
   IfcProject                        →       /GrandBangkok_WestWing
   └─ IfcSite                        →         /Site
      └─ IfcBuilding                 →           /WestWing
         └─ IfcBuildingStorey        →             /Level_01
            ├─ IfcSpace              →               /Lobby        (Xform + mesh)
            ├─ IfcWallStandardCase   →               /Wall_L1_A    (Xform + mesh)
            └─ IfcUnitaryEquipment   →               /AHU_01       (Xform + mesh)
"""


def main() -> None:
    view.banner("PART 2", "What survives BIM → USD", "INTERMEDIATE")
    view.mode_line()

    print("The IFC spatial tree maps 1:1 onto a USD prim hierarchy:\n")
    print(TREE_MAP)

    print("The survival table — what a converter keeps, sometimes keeps, never keeps:\n")
    for grade, rows in sim.SURVIVES:
        print(f"  {grade}")
        for what, why in rows:
            print(f"    • {what:<26}{why}")
        print()

    print("What 'IFC property sets survive' looks like — flattened, namespaced attrs:")
    print('    custom string ifc:GlobalId               = "1cZgN4tVy7QaSePkLw56fJ"')
    print('    custom string ifc:Pset_WallCommon:FireRating = "2HR"')
    print('    custom string ifc:Pset_WallCommon:IsExternal  = "true"\n')

    print("THE load-bearing rule: preserve the IFC GlobalId (GUID) on EVERY prim.")
    print("  • It is the join key that later binds IoT points, CMMS records and the")
    print("    semantics graph to geometry (App 5 does exactly that join).")
    print("  • A converter that drops GUIDs is unusable for twins — full stop.\n")

    print("Honesty note: no ratified IFC-in-USD schema exists yet (buildingSMART and AOUSD")
    print("are collaborating on one), so every pipeline namespaces its own attrs — which is")
    print("why you must CHECK what your converter preserves, not assume it.\n")

    view.generate("In two sentences, why must a BIM-to-USD converter preserve the IFC "
                  "GlobalId on every prim for a building digital twin?", max_tokens=300,
                  title="the GUID join key")

    print("\nTakeaway: meshes, transforms and hierarchy always make it; Psets and materials")
    print("sometimes; intelligence never. Guard the GlobalId — it is the twin's join key.")
    print("Next: which real 2026 pipelines preserve the most.")


if __name__ == "__main__":
    main()
