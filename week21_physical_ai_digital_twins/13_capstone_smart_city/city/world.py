#!/usr/bin/env python3
"""The city inventory — Krung Alto, a 4-district Bangkok-flavored smart city.

Everything carries a stable GlobalId (the App 3/5 lesson at city scale): districts,
road segments, intersections, cameras, substations, buildings. The Grand Bangkok
hotel from Capstone I (App 12) is one building here — a twin nested in a twin.
"""
from __future__ import annotations

DISTRICTS = [
    {"id": "D1", "gid": "0aX9rK1", "name": "Riverside",   "kind": "mixed",       "pop_k": 84},
    {"id": "D2", "gid": "0aX9rK2", "name": "Sukhumvit",   "kind": "commercial",  "pop_k": 121},
    {"id": "D3", "gid": "0aX9rK3", "name": "Old Town",    "kind": "residential", "pop_k": 67},
    {"id": "D4", "gid": "0aX9rK4", "name": "Tech Park",   "kind": "industrial",  "pop_k": 38},
]

# Road segments (the congestion map's atoms). free = free-flow speed km/h.
SEGMENTS = [
    {"id": "S-01", "gid": "1rD5mQ1", "district": "D2", "name": "Sukhumvit Rd km 2–4",  "lanes": 6, "free": 55},
    {"id": "S-02", "gid": "1rD5mQ2", "district": "D2", "name": "Sukhumvit Rd km 4–6",  "lanes": 6, "free": 55},
    {"id": "S-03", "gid": "1rD5mQ3", "district": "D1", "name": "Riverside Blvd",       "lanes": 4, "free": 50},
    {"id": "S-04", "gid": "1rD5mQ4", "district": "D3", "name": "Old Town Loop",        "lanes": 2, "free": 35},
    {"id": "S-05", "gid": "1rD5mQ5", "district": "D4", "name": "Tech Park Arterial",   "lanes": 4, "free": 60},
    {"id": "S-06", "gid": "1rD5mQ6", "district": "D2", "name": "Asoke Interchange",    "lanes": 8, "free": 45},
]

INTERSECTIONS = [
    {"id": "I-11", "gid": "2tG7pW1", "district": "D2", "name": "Sukhumvit × Asoke",   "signal": True},
    {"id": "I-12", "gid": "2tG7pW2", "district": "D2", "name": "Sukhumvit × Thonglor","signal": True},
    {"id": "I-13", "gid": "2tG7pW3", "district": "D1", "name": "Riverside × Bridge",  "signal": True},
    {"id": "I-14", "gid": "2tG7pW4", "district": "D3", "name": "Old Town Gate",       "signal": False},
]

# The camera network — the Smart City Blueprint's raw material. Each camera watches
# one segment or intersection; `model` is the CV detector the RTVI service runs.
CAMERAS = [
    {"id": f"CAM-{n:02d}", "gid": f"3vJ{n:02d}xB", "watch": w, "model": m}
    for n, (w, m) in enumerate([
        ("S-01", "rtdetr"), ("S-01", "rtdetr"), ("S-02", "rtdetr"), ("S-02", "gdino"),
        ("S-03", "rtdetr"), ("S-03", "rtdetr"), ("S-04", "gdino"),  ("S-04", "rtdetr"),
        ("S-05", "rtdetr"), ("S-05", "rtdetr"), ("S-06", "rtdetr"), ("S-06", "gdino"),
        ("I-11", "rtdetr"), ("I-11", "gdino"),  ("I-12", "rtdetr"), ("I-12", "rtdetr"),
        ("I-13", "rtdetr"), ("I-13", "rtdetr"), ("I-14", "gdino"),  ("I-14", "rtdetr"),
        ("S-06", "rtdetr"), ("S-06", "rtdetr"), ("I-11", "rtdetr"), ("I-12", "rtdetr"),
    ], start=1)
]

SUBSTATIONS = [
    {"id": "SUB-A", "gid": "4wK1zN1", "district": "D1", "cap_mw": 40},
    {"id": "SUB-B", "gid": "4wK1zN2", "district": "D2", "cap_mw": 65},
    {"id": "SUB-C", "gid": "4wK1zN3", "district": "D3", "cap_mw": 30},
    {"id": "SUB-D", "gid": "4wK1zN4", "district": "D4", "cap_mw": 55},
]

# Grid-interactive buildings (the CityLearn lesson, App 9 Ch 4). flex_mw = how much
# each can shed/shift on a demand-response call. The hotel is Capstone I's twin.
BUILDINGS = [
    {"id": "B-01", "gid": "5yM3cV1", "district": "D1", "name": "AltoTech Grand Bangkok (App 12 twin)",
     "kind": "hotel",    "peak_mw": 2.4, "flex_mw": 0.5, "storage_mwh": 1.2},
    {"id": "B-02", "gid": "5yM3cV2", "district": "D2", "name": "Asoke Tower offices",
     "kind": "office",   "peak_mw": 3.1, "flex_mw": 0.8, "storage_mwh": 0.0},
    {"id": "B-03", "gid": "5yM3cV3", "district": "D2", "name": "Sukhumvit Mall",
     "kind": "retail",   "peak_mw": 4.2, "flex_mw": 1.1, "storage_mwh": 2.0},
    {"id": "B-04", "gid": "5yM3cV4", "district": "D3", "name": "Old Town Hospital",
     "kind": "hospital", "peak_mw": 2.8, "flex_mw": 0.0, "storage_mwh": 1.5},   # no shed — life safety
    {"id": "B-05", "gid": "5yM3cV5", "district": "D4", "name": "Tech Park data hall",
     "kind": "datacenter","peak_mw": 6.0, "flex_mw": 1.4, "storage_mwh": 3.0},
]


def counts() -> dict:
    return {"districts": len(DISTRICTS), "segments": len(SEGMENTS),
            "intersections": len(INTERSECTIONS), "cameras": len(CAMERAS),
            "substations": len(SUBSTATIONS), "buildings": len(BUILDINGS)}


def by_gid(gid: str) -> dict | None:
    for group in (DISTRICTS, SEGMENTS, INTERSECTIONS, CAMERAS, SUBSTATIONS, BUILDINGS):
        for e in group:
            if e["gid"] == gid:
                return e
    return None
