#!/usr/bin/env python3
"""PART 3 · References, payloads, variants, instancing  [INTERMEDIATE]

The four composition arcs that make a BUILDING-scale stage possible:
REFERENCE composes an asset in · PAYLOAD is a lazy-loadable reference (each
floor its own file, loaded on demand — the #1 scale lever) · VARIANT sets switch
representations (lod: proxy/full — the conceptual bridge to BIM LOD, App 3) ·
INSTANCING makes the 4,000th hotel chair nearly free.

Run:  python demos/step03_refs_payloads_variants.py
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


def main() -> None:
    view.banner("PART 3", "References, payloads, variants, instancing", "INTERMEDIATE")
    sim.usd_mode(REAL)

    box = config.ensure_sandbox() / "usd_step03"
    shutil.rmtree(box, ignore_errors=True)
    box.mkdir(parents=True)

    # 1 · an ASSET with a VARIANT set — chair.usda, authored once, used 4,000 times.
    chair = Usd.Stage.CreateNew(str(box / "chair.usda"))
    c = UsdGeom.Xform.Define(chair, "/Chair")
    Usd.ModelAPI(c.GetPrim()).SetKind("component")
    vs = c.GetPrim().GetVariantSets().AddVariantSet("lod")
    vs.AddVariant("full")
    vs.AddVariant("proxy")
    vs.SetVariantSelection("full")
    with vs.GetVariantEditContext():
        UsdGeom.Mesh.Define(chair, "/Chair/HighPoly")             # 12,400 triangles
    vs.SetVariantSelection("proxy")
    with vs.GetVariantEditContext():
        UsdGeom.Mesh.Define(chair, "/Chair/ProxyBox")             # 12 triangles
    chair.GetRootLayer().Save()

    print("VARIANTS — one asset, switchable representations (like BIM LOD, see App 3):\n")
    for sel in ("proxy", "full"):
        vs.SetVariantSelection(sel)
        kids = [p.GetName() for p in chair.GetPrimAtPath("/Chair").GetChildren()]
        print(f"  lod = {sel:<6} → /Chair children: {kids}")

    # 2 · REFERENCE the asset into a lobby, marked INSTANCEABLE — the geometry
    # lives ONCE in a prototype; each instance is just a transform pointing at it.
    lobby = Usd.Stage.CreateNew(str(box / "lobby.usda"))
    UsdGeom.Xform.Define(lobby, "/Lobby")
    for i in range(1, 4):
        p = lobby.DefinePrim(f"/Lobby/Chair_{i:03d}", "Xform")
        p.GetReferences().AddReference("chair.usda", "/Chair")
        p.SetInstanceable(True)
    lobby.GetRootLayer().Save()

    print("\nREFERENCES + INSTANCING — three chairs, one prototype:\n")
    print(f"  prims traversed: {len(list(lobby.Traverse()))}  "
          "(instances are opaque — their insides live once, in the prototype)")
    print(f"  prototypes on the stage: {len(lobby.GetPrototypes())}")

    print("\n  Why instancing decides whether a building LOADS at all:\n")
    print(f"  {'setup':<32}{'prims':>8}{'tris in memory':>16}   est. memory")
    print("  " + "─" * 76)
    for setup, prims, tris, mem in sim.INSTANCING:
        print(f"  {setup:<32}{prims:>8,}{tris:>16,}   {mem}")

    # 3 · PAYLOAD — a reference the viewer may load LATER. One file per floor →
    # the tower opens instantly; floors stream in on demand.
    floor = Usd.Stage.CreateNew(str(box / "floor01.usda"))
    UsdGeom.Xform.Define(floor, "/Floor")
    UsdGeom.Scope.Define(floor, "/Floor/Rooms")
    for r in range(1, 5):
        UsdGeom.Xform.Define(floor, f"/Floor/Rooms/Room_10{r}")
    floor.GetRootLayer().Save()

    main_ = Usd.Stage.CreateNew(str(box / "main.usda"))
    UsdGeom.Xform.Define(main_, "/Hotel")
    f1 = main_.DefinePrim("/Hotel/Floor_01", "Xform")
    f1.GetPayloads().AddPayload("floor01.usda", "/Floor")
    main_.GetRootLayer().Save()

    print("\nPAYLOADS — open the tower without opening the tower:\n")
    cold = Usd.Stage.Open(str(box / "main.usda"), Usd.Stage.LoadNone)
    print(f"  Open(LoadNone)        → {len(list(cold.Traverse())):>2} prim(s) composed — instant, nothing heavy read")
    cold.Load("/Hotel/Floor_01")
    print(f"  Load('/Hotel/Floor_01') → {len(list(cold.Traverse())):>2} prim(s) composed — the floor streamed in on demand")

    print("\nTakeaway: payload-per-floor is the #1 scale lever, and instancing is the")
    print("difference between a building that loads and one that doesn't. Next: compose")
    print("a whole mini hotel out of every arc you just learned.")


if __name__ == "__main__":
    main()
