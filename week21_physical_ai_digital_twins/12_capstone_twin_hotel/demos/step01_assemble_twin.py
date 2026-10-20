#!/usr/bin/env python3
"""CH 2 · Assemble the twin — the SCENE from BIM to a validated stage  [BEGINNER]

Build AltoTech Grand Bangkok's stage from the inventory in twin/world.py the way
Apps 2–4 taught: discipline layers composed in strength order, a payload per
floor (30 floors, load one), instanced furniture (4,200 chairs, 6 prototypes),
every entity carrying its IFC GlobalId + COBie attrs — then run the
twin-readiness validator before anything is allowed to go live.

Run:  python demos/step01_assemble_twin.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402
from twin import scene, world  # noqa: E402

DIAGRAM = """\
   BIM (IFC, LOD 300)          discipline layers, weakest → strongest
        │                    ┌──────────────────────────────────────┐
        ▼                    │ site < arch < mep < furniture <      │
   inventory (world.py) ───► │ sensors < SESSION (live values win)  │ ──► validator
   GlobalId · COBie · points │ payload/floor · instanced furniture  │     (gate to Ch 3)
                             └──────────────────────────────────────┘
"""


def main() -> None:
    view.banner("CH 2", "Assemble the twin — SCENE, validated", "BEGINNER")
    try:
        from pxr import Usd  # noqa: F401
        print("▣ MODE: REAL — usd-core detected; the same structure composes as genuine USD.\n")
    except ImportError:
        print("▣ MODE: SIM — built-in USD-flavored stage (stdlib, $0). REAL: pip install usd-core\n")

    print("A 3D model is not a twin. First build a SCENE the rest of the stack can bind to:\n")
    print(DIAGRAM)

    inv = world.build_inventory()
    print(f"▣ INVENTORY — {world.HOTEL} ({world.FLOORS} floors, "
          f"{len(world.DETAILED_FLOORS)} modeled in detail)")
    print(f"  {'class':<24}{'count':>6}   examples")
    print("  " + "─" * 66)
    classes: dict[str, list] = {}
    for e in inv.values():
        classes.setdefault(e.ifc_class, []).append(e.name)
    for cls, names in sorted(classes.items()):
        print(f"  {cls:<24}{len(names):>6}   {', '.join(names[:4])}{' …' if len(names) > 4 else ''}")
    ex = inv["AHU-3"]
    print(f"\n  one entity, fully dressed — {ex.name}:")
    print(f"    ifc:GlobalId  {ex.gid}   (22 chars, the join key EVERYTHING binds on)")
    print(f"    cobie:*       {ex.cobie['manufacturer']} {ex.cobie['model']} · "
          f"installed {ex.cobie['installDate']} · warranty→{ex.cobie['warranty']}")
    print(f"    bacnet:ref    {ex.point}")
    print(f"    feeds         {ex.feeds}\n")

    st = scene.assemble(inv)
    print("▣ THE COMPOSED STAGE (floor 5 expanded — the rest stay unloaded payloads)")
    for line in scene.print_tree(st, focus_floor=5):
        print(f"  {line}")
    print()

    print("▣ TWIN-READINESS VALIDATOR — run BEFORE binding anything live")
    print(f"  {'check':<44}{'status':<7}detail")
    print("  " + "─" * 82)
    warns = 0
    for check, status, detail in scene.validate(st, inv):
        mark = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}[status]
        warns += status != "PASS"
        print(f"  {check:<44}[{status}] {mark} {detail}")
    print(f"\n  verdict: stage is twin-ready with {warns} warning(s) — both are REAL handover")
    print("  problems (a never-re-tagged sensor, VAVs missing serial/warranty). Fix them in")
    print("  the BIM/COBie source, not in the twin — the twin only ever mirrors.\n")

    print("run it for real:")
    print("  $ pip install usd-core        # genuine OpenUSD, no GPU (App 2)")
    print("  $ pip install ifcopenshell    # the IFC-first conversion path (App 3)\n")

    print("Takeaway: SCENE = layers + payloads + instancing + identity. The two WARNs above")
    print("are why twins fail — missing DATA, not missing geometry. Next: make it live.")


if __name__ == "__main__":
    main()
