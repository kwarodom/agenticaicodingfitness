#!/usr/bin/env python3
"""PART 1 · Stage · prim · attribute · relationship  [BEGINNER]

USD's scene graph in one hotel room: a STAGE holds a tree of PRIMS (Xform/Scope/
Mesh) at BIM-shaped paths; prims carry ATTRIBUTES (displayColor, a custom
`ifc:GlobalId`) and RELATIONSHIPS (a sensor pointing at the equipment it
instruments), plus `kind` and `purpose` metadata — and the two classic gotchas,
metersPerUnit and upAxis. REAL with `pip install usd-core`, else the mini-USD
engine in sim.py.

Run:  python demos/step01_stage_prims.py
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
    from pxr import Sdf, Usd, UsdGeom  # noqa: E402  — real OpenUSD (pip install usd-core)
    REAL = True
except ImportError:
    from sim import Sdf, Usd, UsdGeom  # noqa: E402  — the teaching engine in sim.py
    REAL = False

ROOM = "/Site/Building/Storey_03/Rooms/Room_301"


def rgb(color) -> str:
    return "(" + ", ".join(f"{c:.2f}" for c in color[0]) + ")" if color else "(none)"


def main() -> None:
    view.banner("PART 1", "Stage · prim · attribute · relationship", "BEGINNER")
    sim.usd_mode(REAL)

    box = config.ensure_sandbox() / "usd_step01"
    shutil.rmtree(box, ignore_errors=True)
    box.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(box / "room301.usda"))

    print("GOTCHA — units & axes. A brand-new stage defaults to:")
    print(f"  metersPerUnit = {UsdGeom.GetStageMetersPerUnit(stage)} (centimeters!)   "
          f"upAxis = {UsdGeom.GetStageUpAxis(stage)}")
    print("  Revit thinks in FEET and Z-up. Skip this and your hotel imports 100x too")
    print("  small, lying on its side. Set both explicitly, every time:")
    UsdGeom.SetStageMetersPerUnit(stage, 0.3048)                  # 1 unit = 1 foot
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    print(f"  → metersPerUnit = {UsdGeom.GetStageMetersPerUnit(stage)} (feet)   "
          f"upAxis = {UsdGeom.GetStageUpAxis(stage)}\n")

    # PRIMS — the tree. Xform = a placeable node, Scope = a pure folder, Mesh = geometry.
    site = UsdGeom.Xform.Define(stage, "/Site")
    bld = UsdGeom.Xform.Define(stage, "/Site/Building")
    sty = UsdGeom.Xform.Define(stage, "/Site/Building/Storey_03")
    UsdGeom.Scope.Define(stage, "/Site/Building/Storey_03/Rooms")
    UsdGeom.Xform.Define(stage, ROOM)
    wall = UsdGeom.Mesh.Define(stage, ROOM + "/Wall_W1")
    fcu = UsdGeom.Xform.Define(stage, ROOM + "/FCU_301")          # fan-coil unit
    sensor = stage.DefinePrim(ROOM + "/TempSensor_301", "Xform")

    # ATTRIBUTES — values on a prim: a built-in (displayColor) and a custom one
    # (the IFC GUID — the join key that later binds live telemetry to this wall).
    wall.CreateDisplayColorAttr([(0.85, 0.84, 0.78)])
    wall.GetPrim().CreateAttribute("ifc:GlobalId", Sdf.ValueTypeNames.String).Set(
        "2O2Fr$t4X7Zf8NOew3FLKI")
    # RELATIONSHIP — pure meaning, no geometry: the sensor POINTS AT what it measures.
    rel = sensor.CreateRelationship("twin:instruments")
    rel.AddTarget(fcu.GetPrim().GetPath())
    # KIND — assembly (building blocks) vs component (leaf equipment): lets tools
    # select at the right granularity ("pick the floor", "pick the FCU").
    for prim in (site.GetPrim(), bld.GetPrim(), sty.GetPrim()):
        Usd.ModelAPI(prim).SetKind("assembly")
    Usd.ModelAPI(fcu.GetPrim()).SetKind("component")
    # PURPOSE — which representation a viewport draws: proxy / render / guide.
    UsdGeom.Imageable(wall.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.render)
    stage.GetRootLayer().Save()

    print("The stage, composed (this tree is what `usdview` shows):\n")
    for prim in stage.Traverse():
        depth = str(prim.GetPath()).count("/") - 1
        kind = Usd.ModelAPI(prim).GetKind()
        print(f"  {'  ' * depth}{prim.GetName():<16} {str(prim.GetTypeName()):<6}"
              f"{('  [' + str(kind) + ']') if kind else ''}")

    color = wall.GetPrim().GetAttribute("primvars:displayColor").Get()
    gid = wall.GetPrim().GetAttribute("ifc:GlobalId").Get()
    print("\nAttributes on Wall_W1:")
    print(f"  primvars:displayColor = {rgb(color)}        ← built-in: how it draws")
    print(f"  ifc:GlobalId          = {gid!r}   ← custom: the BIM join key (Apps 3, 5)")
    print(f"Relationship:  TempSensor_301 —twin:instruments→ {rel.GetTargets()[0]}\n")

    head = (box / "room301.usda").read_text().splitlines()
    print("room301.usda — USD scene description is PLAIN TEXT (first 12 lines):")
    for ln in head[:12]:
        print("  │ " + ln)

    print("\nTakeaway: stage = the composed scene, prim = a node, attribute = a value,")
    print("relationship = a pointer — and always author metersPerUnit + upAxis. Next:")
    print("layers, and how BIM disciplines federate without overwriting each other.")


if __name__ == "__main__":
    main()
