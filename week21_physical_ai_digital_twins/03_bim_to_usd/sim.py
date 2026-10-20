#!/usr/bin/env python3
"""BIM → USD simulator — a mock-IFC hotel wing, converter tables, twin-readiness rules.

Pure stdlib. The demos import this for:
  • LOD_TABLE / SURVIVES / PIPELINES / SCORECARD — the teaching tables (Ch 2–4)
  • MOCK_IFC + flatten()                         — a ~2-storey mock-IFC hotel wing (Ch 5)
  • lint helpers                                 — GUID coverage, COBie completeness, orphans
  • installed_models / tok_s / stream_generate   — the standard sim LLM interface
"""
from __future__ import annotations

import time

# ── Ch 2 · BIMForum LOD table (Level of DEVELOPMENT, not detail) ──────────────
# (lod, name, what the geometry means, what it means for a twin)
LOD_TABLE = [
    ("100", "symbol / massing",       "approximate only — area, height, volume, location, orientation",
     "concept studies; too coarse to bind live data to"),
    ("200", "generic placeholder",    "approximate size, shape, location",
     "enough to anchor sensors and zones in a twin"),
    ("300", "specific, as designed",  "measurable directly from the model; project origin defined",
     "the twin sweet spot for architecture"),
    ("350", "300 + interfaces",       "cross-trade supports, hangers, clearances",
     "the clash-coordination level; right for maintained MEP"),
    ("400", "fabrication / assembly", "fabrication- and installation-level detail",
     "polygon bloat — actively hurts a real-time twin"),
    ("500", "field-verified as-built","an ATTESTATION — BIMForum deliberately defines NO LOD 500 geometry",
     "what a twin needs is LOD-500-grade DATA (COBie)"),
]

# ── Ch 3 · what survives a BIM → USD conversion ───────────────────────────────
SURVIVES = [
    ("ALWAYS", [
        ("tessellated meshes",   "the triangles themselves — every converter emits these"),
        ("transforms",           "position / rotation / scale of every element"),
        ("hierarchy",            "the IFC spatial tree maps 1:1 onto the USD prim tree"),
    ]),
    ("USUALLY (converter-dependent)", [
        ("materials",            "approximated as UsdPreviewSurface / MDL — not the BIM material"),
        ("IFC property sets",    "flattened to namespaced attrs, e.g. ifc:Pset_WallCommon:FireRating"),
        ("Revit parameters",     "survive as flat metadata when the exporter carries them"),
    ]),
    ("NEVER", [
        ("parametric intelligence", "walls stop knowing they're walls — a wall becomes a mesh"),
        ("hosted relationships",    "door-hosted-in-wall, fixture-on-ceiling logic is gone"),
        ("families / types as logic","USD is scene description, not BIM authoring"),
    ]),
]

# ── Ch 4 · the five real pipelines of the post-connector era (2026) ───────────
# (rank, path, one-line verdict)
PIPELINES = [
    ("1", "Revit → IFC4 export → IfcOpenShell-based converter",
     "open-source Python (ifcopenshell.org) — the most metadata-faithful open path"),
    ("2", "Omniverse CAD Converter (Kit extension)",
     "built on Tech Soft 3D HOOPS Exchange; reads IFC directly"),
    ("3", "Bentley iTwin → LumenRT for Omniverse",
     "infrastructure-grade, but a heavier platform commitment"),
    ("4", "Datasmith → Unreal → USD",
     "BIM params survive as Datasmith metadata; lossy — use when your frontend is Unreal"),
    ("5", "FBX / glTF export",
     "geometry only — WRONG for twins (identity and data stripped)"),
]

# Simulated preservation scorecard: the SAME hotel wing pushed through paths 1, 4, 5.
# (label, geometry %, psets %, GUIDs kept?)  — illustrative sim numbers.
SCORECARD = [
    ("Path 1 · IFC4 + IfcOpenShell",     100, 96, "YES — ifc:GlobalId on every prim"),
    ("Path 4 · Datasmith → Unreal → USD", 98, 61, "partial — buried in Datasmith metadata, not on prims"),
    ("Path 5 · FBX / glTF",              100,  0, "NO — identity stripped; unusable for a twin"),
]

# ── Ch 5 · the mock-IFC hotel wing (2 storeys, spaces, AHU + VAVs + walls) ────
COBIE_FIELDS = ("manufacturer", "model", "serial", "installDate", "warranty", "zone")
MAINTAINED = {"IfcUnitaryEquipment": "AHU", "IfcAirTerminalBox": "VAV"}

def _n(typ, name, gid, children=None, psets=None, cobie=None):
    return {"type": typ, "name": name, "global_id": gid,
            "children": children or [], "psets": psets or {}, "cobie": cobie or {}}

MOCK_IFC = _n("IfcProject", "GrandBangkok_WestWing", "2O2Fr9qzb5FQyLnfeZwGmN", [
  _n("IfcSite", "Site", "0jWkQ3vTz1JeXHcAyR8pDs", [
    _n("IfcBuilding", "WestWing", "1XhVu6mYzC0O5qgtBvN4kA", [
      _n("IfcBuildingStorey", "Level_01", "3fA9dPq2j8BxKtMwZ7yHcE", [
        _n("IfcSpace", "Lobby",     "0sQeL5nRw4TgUvJmXc21bF"),
        _n("IfcSpace", "Guest_101", "2mKdT8rWx6PbYzGqAj93eH"),
        _n("IfcWallStandardCase", "Wall_L1_A", "1cZgN4tVy7QaSePkLw56fJ",
           psets={"Pset_WallCommon": {"FireRating": "2HR", "IsExternal": "true"}}),
        _n("IfcUnitaryEquipment", "AHU_01", "3vBxE7wUz9RcTfHnMy48gK",
           psets={"Pset_UnitaryEquipmentTypeCommon": {"AirFlowRate_m3s": "8.5"}},
           cobie={"manufacturer": "TropiCool Air", "model": "AHU-40k",
                  "serial": "TC-AHU-2024-118", "installDate": "2024-11-02",
                  "warranty": "2029-11-02", "zone": "Zone_L1_West"}),
      ]),
      _n("IfcBuildingStorey", "Level_02", "0pHjR6sXw3NdVaKuFz75mC", [
        _n("IfcSpace", "Guest_201", "1qLfU9vYx5ScWbJnGk32dM"),
        _n("IfcSpace", "Guest_202", "2tNhW4xZy8UdXcKpHm67eP"),
        _n("IfcWallStandardCase", "Wall_L2_A", "3wCkF8uTz6QeYgLrJn19hR",
           psets={"Pset_WallCommon": {"FireRating": "2HR", "IsExternal": "true"}}),
        _n("IfcAirTerminalBox", "VAV_2_01", "0yDmG5vWx2ReZhNsKp84jT",
           psets={"Pset_AirTerminalBoxTypeCommon": {"AirFlowRate_m3s": "0.35"}},
           cobie={"manufacturer": "TropiCool Air", "model": "VAV-R2", "serial": "",
                  "installDate": "", "warranty": "", "zone": "Zone_L2_Guest"}),
        _n("IfcAirTerminalBox", "VAV_2_02", "1zEnH7wXy4SfAiPtLq95kV",
           psets={"Pset_AirTerminalBoxTypeCommon": {"AirFlowRate_m3s": "0.35"}},
           cobie={"manufacturer": "TropiCool Air", "model": "VAV-R2", "serial": "",
                  "installDate": "", "warranty": "", "zone": "Zone_L2_Guest"}),
      ]),
      # deliberately misplaced: a space parented to the building, not to a storey.
      _n("IfcSpace", "Roof_Plant", "2aGpJ9xYz6TgBjQuMr07lW"),
    ]),
  ]),
])


def flatten(node=None, parent_path="", parent_type=""):
    """Depth-first walk → (usd_path, node, parent_type) for every element."""
    node = node or MOCK_IFC
    path = f"{parent_path}/{node['name']}"
    yield path, node, parent_type
    for child in node["children"]:
        yield from flatten(child, path, node["type"])


# ── twin-readiness lint helpers ───────────────────────────────────────────────
def guid_coverage() -> tuple[int, int]:
    """(elements with a GlobalId, total elements) in the converted stage."""
    nodes = [n for _, n, _ in flatten()]
    return sum(1 for n in nodes if n.get("global_id")), len(nodes)


def cobie_report() -> dict:
    """Per maintained asset class: {'AHU': (pct, count, [missing fields])}."""
    out = {}
    for _, n, _ in flatten():
        cls = MAINTAINED.get(n["type"])
        if not cls:
            continue
        filled, count, missing = out.get(cls, (0, 0, set()))
        for f in COBIE_FIELDS:
            if n["cobie"].get(f):
                filled += 1
            else:
                missing.add(f)
        out[cls] = (filled, count + 1, missing)
    return {cls: (round(100 * filled / (6 * count)), count, sorted(missing))
            for cls, (filled, count, missing) in out.items()}


def orphaned_spaces() -> list[str]:
    """IfcSpace elements not parented to an IfcBuildingStorey."""
    return [n["name"] for _, n, ptype in flatten()
            if n["type"] == "IfcSpace" and ptype != "IfcBuildingStorey"]


# ── the standard sim LLM interface (installed_models / tok_s / stream_generate) ─
_MODELS = ["nemotron-3-nano:30b-a3b", "gemma4:12b"]
_TOK = {"nemotron-3-nano:30b-a3b": 54.0, "gemma4:12b": 38.0}


def installed_models() -> list[str]:
    return _MODELS


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40.0))


_CANNED = ("[simulated] The IFC GlobalId is the one identifier that already exists in every "
           "system that touches the building — the BIM model, the CMMS, the BMS point list, the "
           "semantics graph. Preserve it as ifc:GlobalId on every USD prim and geometry, "
           "telemetry, work orders and graph nodes all join on the same key; drop it and the "
           "stage is just a pretty mesh no data can ever bind to.")


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)
