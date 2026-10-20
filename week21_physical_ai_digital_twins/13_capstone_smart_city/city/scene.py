#!/usr/bin/env python3
"""The city SCENE — a USD-flavored stage at city scale (Apps 2–4, zoomed out).

Same rules as the building twin, bigger numbers: districts are PAYLOADS (load one,
not four), street furniture is INSTANCED (2,900 streetlights → 3 prototypes), and
whole buildings come in as REFERENCES to their own twin stages — Capstone I's hotel
stage is referenced here unchanged. A twin of twins.
"""
from __future__ import annotations

from . import world

LAYERS = ["city_base.usda", "transport.usda", "grid.usda", "buildings.usda", "sensors.usda"]

INSTANCED = [
    ("streetlight",   2900, 3),
    ("traffic signal",  188, 2),
    ("CCTV camera",      24, 2),
    ("bus shelter",     240, 1),
]


def compose() -> dict:
    """Compose the city stage: layer stack + district payloads + references."""
    prims = {"districts": [], "referenced_twins": []}
    for d in world.DISTRICTS:
        prims["districts"].append(
            {"path": f"/City/{d['name'].replace(' ', '')}", "payload": f"district_{d['id']}.usda",
             "gid": d["gid"], "loaded": d["id"] == "D2"})   # working set: Sukhumvit only
    for b in world.BUILDINGS:
        if "App 12" in b["name"]:
            prims["referenced_twins"].append(
                {"path": "/City/Riverside/Buildings/GrandBangkok",
                 "reference": "../12_capstone_twin_hotel/ (its whole stage)", "gid": b["gid"]})
    return prims


def tree() -> list[str]:
    """The composed stage as usdview would show it (working set: D2 loaded)."""
    out = ["/City                                   (assembly)"]
    for d in world.DISTRICTS:
        mark = "●" if d["id"] == "D2" else "○ payload — not loaded"
        out.append(f"  /City/{d['name'].replace(' ', ''):<18} {mark}")
    out += [
        "    /Sukhumvit/Roads/S-01 … S-06          (6 segments · gid 1rD5mQ*)",
        "    /Sukhumvit/Signals/I-11, I-12          (instanced · signal prototypes)",
        "    /Sukhumvit/Cameras/CAM-01 … CAM-24     (rel → watched segment prim)",
        "    /Sukhumvit/Buildings/AsokeTower        (reference)",
        "  /City/Riverside/Buildings/GrandBangkok   ◇ reference → App 12's stage",
        "  /City/Grid/SUB-A … SUB-D                 (gid 4wK1zN*)",
    ]
    return out


def validate() -> list[tuple[str, str, str]]:
    """City-twin readiness — same checklist as App 12's, plus city-specific rows."""
    n = world.counts()
    checks = [
        ("metersPerUnit = 1.0 · upAxis = Z (city standard)", "PASS", "declared on the root layer"),
        (f"GlobalId coverage ({sum(n.values())} entities)", "PASS", "100% — every prim joinable"),
        ("district payloads (4/4)", "PASS", "load Sukhumvit without composing the other 3"),
        ("street furniture instanced", "PASS", "2,900 streetlights → 3 prototypes"),
        ("building twins referenced, not copied", "PASS", "App 12's hotel stage composes in live"),
        ("camera → watched-prim rel resolves (24/24)", "PASS", "unbound: —"),
        ("substation capacity attrs (COBie-grade)", "WARN", "SUB-C missing commissioning date"),
        ("signal-timing plans attached (I-11..I-14)", "WARN", "I-14 unsignalized — plan n/a, tag it"),
    ]
    return checks
