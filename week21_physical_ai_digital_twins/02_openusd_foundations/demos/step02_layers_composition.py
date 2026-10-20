#!/usr/bin/env python3
"""PART 2 · Layers & composition — how disciplines federate  [INTERMEDIATE]

USD's superpower is NON-DESTRUCTIVE layering: every discipline keeps its own
file (architecture.usda, mep.usda, …), a building file just SUBLAYERS them, and
layer ORDER = opinion STRENGTH. A stronger layer overrides a weaker one without
touching it — remove the override and the original shines through. Exactly how
BIM disciplines federate, but native to the format.

Run:  python demos/step02_layers_composition.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import sim  # noqa: E402
import view  # noqa: E402

try:
    from pxr import Sdf, Usd, UsdGeom  # noqa: E402
    REAL = True
except ImportError:
    from sim import Sdf, Usd, UsdGeom  # noqa: E402
    REAL = False

WALL = "/Site/Building/Storey_01/Wall_W1"
COLOR = "primvars:displayColor"

STACK = """\
   building.usda            ← the federation file: just a sublayer LIST
     ├─ session layer       (strongest — live runtime values, RAM only)
     ├─ ops.usda            (stronger  — operations' overrides)
     ├─ architecture.usda   (weaker    — walls, rooms, finishes)
     └─ mep.usda            (weaker    — AHUs, ducts, pipes)
"""


def rgb(color) -> str:
    return "(" + ", ".join(f"{c:.2f}" for c in color[0]) + ")" if color else "(none)"


def main() -> None:
    view.banner("PART 2", "Layers & composition — disciplines federate", "INTERMEDIATE")
    sim.usd_mode(REAL)

    box = config.ensure_sandbox() / "usd_step02"
    shutil.rmtree(box, ignore_errors=True)
    box.mkdir(parents=True)

    # 1 · each discipline authors its OWN file — nobody edits anyone else's model.
    arch = Usd.Stage.CreateNew(str(box / "architecture.usda"))
    UsdGeom.Mesh.Define(arch, WALL).CreateDisplayColorAttr([(0.85, 0.84, 0.78)])
    arch.GetRootLayer().Save()

    mep = Usd.Stage.CreateNew(str(box / "mep.usda"))
    UsdGeom.Xform.Define(mep, "/Site/Building/Storey_01/AHU_01")
    mep.GetRootLayer().Save()

    # 2 · operations wants that wall RED (say, a maintenance flag). It authors an
    # *over* — an opinion with no definition — in its OWN layer. architecture.usda
    # is never opened for writing, let alone edited.
    ops = Usd.Stage.CreateNew(str(box / "ops.usda"))
    over = ops.OverridePrim(WALL)
    over.CreateAttribute(COLOR, Sdf.ValueTypeNames.Color3fArray).Set([(0.90, 0.15, 0.10)])
    ops.GetRootLayer().Save()

    print("Three discipline files on disk; one building composes them:\n")
    print(STACK)

    # 3 · federate: building.usda is nothing but an ordered sublayer list.
    bld = Usd.Stage.CreateNew(str(box / "building.usda"))
    bld.GetRootLayer().subLayerPaths = ["ops.usda", "architecture.usda", "mep.usda"]
    wall = bld.GetPrimAtPath(WALL)
    print(f"  composed Wall_W1 color = {rgb(wall.GetAttribute(COLOR).Get())}   ← ops.usda wins (stronger)")

    print("\n  …but BOTH opinions still exist — nothing was destroyed:")
    for f in ("ops.usda", "architecture.usda"):
        lyr = Usd.Stage.Open(str(box / f))
        c = lyr.GetPrimAtPath(WALL).GetAttribute(COLOR).Get()
        print(f"    {f:<20} its own opinion: {rgb(c)}")

    # 4 · drop the override → the weaker opinion shines through, untouched.
    bld.GetRootLayer().subLayerPaths = ["architecture.usda", "mep.usda"]
    wall = bld.GetPrimAtPath(WALL)
    print(f"\n  remove ops.usda from the stack →")
    print(f"  composed Wall_W1 color = {rgb(wall.GetAttribute(COLOR).Get())}   ← architecture shines through")

    # 5 · the SESSION layer — the strongest layer of all, lives in RAM, never saved.
    # This is where App 5 will write live telemetry (temperatures, setpoints).
    bld.SetEditTarget(bld.GetSessionLayer())
    wall = bld.GetPrimAtPath(WALL)
    wall.CreateAttribute("twin:liveTempC", Sdf.ValueTypeNames.Float).Set(23.5)
    arch_alone = Usd.Stage.Open(str(box / "architecture.usda"))    # keep the stage alive!
    ondisk = arch_alone.GetPrimAtPath(WALL).GetAttribute("twin:liveTempC").Get()
    print(f"\n  session layer: Wall_W1.twin:liveTempC = {wall.GetAttribute('twin:liveTempC').Get()}"
          f"   (composed, live)")
    print(f"  architecture.usda on disk has that attribute: {ondisk!r}   ← runtime never pollutes source\n")
    bld.GetRootLayer().Save()

    print("Takeaway: layer order = opinion strength; overrides are opinions, not edits —")
    print("that's THE USD superpower, and it is exactly how BIM disciplines federate.")
    print("Next: references, payloads, variants and instancing — composing whole assets.")


if __name__ == "__main__":
    main()
