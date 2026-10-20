#!/usr/bin/env python3
"""PART 4 · Compose a mini hotel  [ADVANCED]

Everything at once: a site + building stage, two floors as PAYLOADS (each its
own file), INSTANCED furniture referenced from one chair asset, and a Sensors
scope carrying IFC GlobalIds — then the composed tree (what usdview shows), and
the honest contrast: FLATTEN for handoff, keep the LIVE COMPOSITION for the twin.

Run:  python demos/step04_mini_hotel.py
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

SENSORS = [                       # (prim name, IFC GlobalId — the join key for App 5)
    ("TempSensor_101", "0kF2qazT91xhVs3D7pLwYc"),
    ("TempSensor_201", "1zXv5rBn4dQe8mKjW2oPtH"),
    ("PowerMeter_MDB", "3gT7yCw2uVr0aNs9bE4xQd"),
]


def main() -> None:
    view.banner("PART 4", "Compose a mini hotel", "ADVANCED")
    sim.usd_mode(REAL)

    box = config.ensure_sandbox() / "usd_step04"
    shutil.rmtree(box, ignore_errors=True)
    box.mkdir(parents=True)

    # 1 · the furniture asset — authored once, instanced everywhere.
    chair = Usd.Stage.CreateNew(str(box / "chair.usda"))
    ch = UsdGeom.Xform.Define(chair, "/Chair")
    Usd.ModelAPI(ch.GetPrim()).SetKind("component")
    UsdGeom.Mesh.Define(chair, "/Chair/Geom")
    chair.GetRootLayer().Save()

    # 2 · one file per floor (the payload target), rooms + instanced chairs inside.
    for n in (1, 2):
        fs = Usd.Stage.CreateNew(str(box / f"floor0{n}.usda"))
        UsdGeom.Xform.Define(fs, "/Floor")
        UsdGeom.Scope.Define(fs, "/Floor/Rooms")
        for r in range(1, 4):
            room = f"/Floor/Rooms/Room_{n}0{r}"
            UsdGeom.Xform.Define(fs, room)
            for k in (1, 2):
                seat = fs.DefinePrim(f"{room}/Chair_{k}", "Xform")
                seat.GetReferences().AddReference("chair.usda", "/Chair")
                seat.SetInstanceable(True)
        fs.GetRootLayer().Save()

    # 3 · the sensors DISCIPLINE layer — GlobalIds are how App 5 binds telemetry.
    sens = Usd.Stage.CreateNew(str(box / "sensors.usda"))
    UsdGeom.Scope.Define(sens, "/Site/Building/Sensors")
    for name, gid in SENSORS:
        p = sens.DefinePrim(f"/Site/Building/Sensors/{name}", "Xform")
        p.CreateAttribute("ifc:GlobalId", Sdf.ValueTypeNames.String).Set(gid)
    sens.GetRootLayer().Save()

    # 4 · the hotel: units set, kinds set, floors payloaded, sensors sublayered.
    hotel = Usd.Stage.CreateNew(str(box / "hotel.usda"))
    UsdGeom.SetStageMetersPerUnit(hotel, 0.3048)
    UsdGeom.SetStageUpAxis(hotel, UsdGeom.Tokens.z)
    Usd.ModelAPI(UsdGeom.Xform.Define(hotel, "/Site").GetPrim()).SetKind("assembly")
    Usd.ModelAPI(UsdGeom.Xform.Define(hotel, "/Site/Building").GetPrim()).SetKind("assembly")
    for n in (1, 2):
        fl = hotel.DefinePrim(f"/Site/Building/Floor_0{n}", "Xform")
        fl.GetPayloads().AddPayload(f"floor0{n}.usda", "/Floor")
        Usd.ModelAPI(fl).SetKind("assembly")
    hotel.GetRootLayer().subLayerPaths = ["sensors.usda"]
    hotel.GetRootLayer().Save()

    print("The composed stage — this tree is what usdview (or Omniverse) shows:\n")
    for prim in hotel.Traverse():
        depth = str(prim.GetPath()).count("/") - 1
        kind = Usd.ModelAPI(prim).GetKind()
        tag = ("  [" + str(kind) + "]") if kind else ""
        tag += "  ⟲ instance" if prim.IsInstanceable() else ""
        print(f"  {'  ' * depth}{prim.GetName():<16} {str(prim.GetTypeName()):<6}{tag}")

    n_prims = len(list(hotel.Traverse()))
    n_files = len(list(box.glob("*.usda")))
    print(f"\n  {n_prims} prims composed live from {n_files} files · "
          f"{len(hotel.GetPrototypes())} chair prototype shared by 12 instances")
    print("  Sensor GlobalIds carried on the stage (the App 5 join key):")
    for name, gid in SENSORS:
        print(f"    /Site/Building/Sensors/{name:<15} ifc:GlobalId = {gid}")

    # 5 · flatten vs live composition — two artifacts, two jobs.
    hotel.Export(str(box / "hotel_flat.usda"))
    flat = Usd.Stage.Open(str(box / "hotel_flat.usda"))
    live_b = (box / "hotel.usda").stat().st_size
    flat_b = (box / "hotel_flat.usda").stat().st_size
    print(f"\nFLATTEN vs LIVE COMPOSITION:")
    print(f"  hotel.usda       {live_b:>7,} bytes — a sublayer list + 2 payload arcs; edits in any")
    print(f"                                discipline file flow through; ops can override non-destructively")
    print(f"  hotel_flat.usda  {flat_b:>7,} bytes — {len(list(flat.Traverse()))} prims baked into ONE self-contained file:")
    print(f"                                perfect handoff copy, but the layer history is GONE —")
    print(f"                                every future edit would be destructive")

    print("\nTakeaway: flatten for handoff, composition for the living twin. This stage is")
    print("the SCENE layer of Digital Twin = SCENE + STATE + SIMULATION + AGENTS — App 3")
    print("fills it from real BIM (IFC, LOD, GUIDs), App 4 scales it, App 5 makes it live.")


if __name__ == "__main__":
    main()
