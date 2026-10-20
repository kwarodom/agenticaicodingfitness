#!/usr/bin/env python3
"""Staff-copilot simulator — a mini building twin the demos query offline.

The copilot's four TOOL stores, all pure stdlib and deterministic:
  • GRAPH — Brick/RealEstateCore-flavoured triples (what feeds what, where things are)
  • TSDB  — 24 h of telemetry for the west-ballroom fault scenario
  • CMMS  — work-order history, nearby PMs, technicians on shift
  • DOCS  — O&M-manual snippets for document RAG

Plus the dispatch problem (8 work orders × 3 technicians) for the cuOpt chapter,
an occupancy feed + schedule what-if for the Metropolis chapter, and the standard
installed_models() / tok_s() / stream_generate() trio (canned = the grounded
ballroom answer).
"""
from __future__ import annotations

import time

# ── 1) semantics graph (Brick / RealEstateCore / ASHRAE 223P-flavoured triples) ──
GRAPH = [
    # the west-ballroom scenario
    ("AHU-2",         "feeds",      "VAV-1-12"),
    ("VAV-1-12",      "serves",     "West Ballroom"),
    ("West Ballroom", "isPartOf",   "Floor 1"),
    ("West Ballroom", "hasRoomRef", "B-102"),
    ("VAV-1-12",      "locatedIn",  "ceiling void above B-102"),
    ("VAV-1-12",      "accessVia",  "service corridor S-1 (BOH badge · ladder required)"),
    ("VAV-1-12",      "hasPart",    "damper actuator ACT-24M"),
    ("AHU-2",         "locatedIn",  "roof plant room R-2"),
    # the classic one-liner: all VAVs fed by AHU-3 on floor 5
    ("AHU-3", "feeds", "VAV-5-01"), ("VAV-5-01", "isPartOf", "Floor 5"),
    ("AHU-3", "feeds", "VAV-5-02"), ("VAV-5-02", "isPartOf", "Floor 5"),
    ("AHU-3", "feeds", "VAV-5-03"), ("VAV-5-03", "isPartOf", "Floor 5"),
    ("AHU-3", "feeds", "VAV-5-04"), ("VAV-5-04", "isPartOf", "Floor 5"),
    ("AHU-3", "feeds", "VAV-4-09"), ("VAV-4-09", "isPartOf", "Floor 4"),
]


def graph_query(s=None, p=None, o=None) -> list[tuple[str, str, str]]:
    """Match triples with None as wildcard — one query, not a BMS spelunking session."""
    return [t for t in GRAPH
            if (s is None or t[0] == s) and (p is None or t[1] == p)
            and (o is None or t[2] == o)]


# ── 2) time-series historian (last 24 h, hour 0 = 24 h ago) ──────────────────────
_STUCK_H = 6      # the damper stuck ~06:00 in the window (hour index 6)


def telemetry(point: str, hours: int = 24) -> list[tuple[int, float]]:
    """Deterministic hourly series for the fault scenario."""
    out = []
    for h in range(hours):
        if point == "VAV-1-12.damper_pos_pct":
            v = float([42, 55, 61, 48, 39, 52][h % 6]) if h < _STUCK_H else 12.0
        elif point == "West Ballroom.zone_temp_c":
            v = 23.1 if h <= _STUCK_H else min(26.8, round(23.1 + (h - _STUCK_H) * 0.31, 1))
        elif point == "AHU-2.supply_air_temp_c":
            v = round(12.8 + (h % 5) * 0.1, 1)
        else:
            v = 0.0
        out.append((h, v))
    return out


# ── 3) CMMS — history, technicians, nearby PMs ────────────────────────────────────
WO_HISTORY = {
    "VAV-1-12": [
        ("WO-7141", "2025-04-18", "damper actuator replaced (ACT-24M) — intermittent sticking"),
        ("WO-6890", "2024-11-02", "zone temp sensor recalibrated"),
    ],
    "AHU-2": [("WO-7302", "2026-05-30", "filter bank changed (PM)")],
}

TECHNICIANS = [
    {"name": "Somchai", "skills": ["hvac", "controls"],            "shift": "06:00–15:00"},
    {"name": "Priya",   "skills": ["electrical", "lighting", "controls"], "shift": "06:00–15:00"},
    {"name": "Marco",   "skills": ["plumbing", "hvac"],            "shift": "07:00–16:00"},
]

# PMs on (or next to) the same access path as the ballroom fix — batch the visit.
NEARBY_PM = [
    ("PM-2101", "AHU-2 filter ΔP check", "roof plant room R-2 (feeds the same zone)", "due in 6 days", 20),
    ("PM-2088", "corridor S-1 emergency-light test", "service corridor S-1 (the access path itself)", "due in 11 days", 10),
]


def cmms_history(asset: str) -> list[tuple[str, str, str]]:
    return WO_HISTORY.get(asset, [])


def on_shift(skill: str) -> list[dict]:
    return [t for t in TECHNICIANS if skill in t["skills"]]


# ── 4) O&M-manual document store (RAG over snippets) ─────────────────────────────
DOCS = {
    "vav damper actuator stuck": (
        "VAV terminal O&M manual §4.2 — damper actuator (ACT-24M): if the damper holds "
        "position while the control signal modulates, replace the actuator and recalibrate "
        "min/max stroke after fitting. LOCKOUT/TAGOUT: isolate the 24 V control circuit at "
        "panel CP-1-B before removing the linkage."),
    "ahu filter change": (
        "AHU O&M manual §7.1 — change the filter bank when ΔP exceeds 250 Pa; log the "
        "before/after ΔP in the CMMS."),
    "chiller vibration": (
        "Chiller O&M manual §3.4 — vibration above 7 mm/s RMS at the compressor mount "
        "warrants a bearing inspection; schedule within 14 days."),
}


def doc_search(query: str) -> tuple[str, str]:
    """Tiny keyword RAG: return (doc_key, snippet) with the best word overlap."""
    q = set(query.lower().split())
    best, score = ("", "no matching manual section found"), 0
    for k, snip in DOCS.items():
        s = len(q & set(k.split()))
        if s > score:
            best, score = (k, snip), s
    return best


# ── dispatch problem for the cuOpt chapter ────────────────────────────────────────
# (id, task, skill, site (x, y) on a walking grid, service_min, window minutes after 08:00)
DISPATCH_WOS = [
    ("WO-7411", "VAV-1-12 damper actuator swap (ballroom)", "hvac",       (2, 1), 45, (0, 240)),
    ("WO-7404", "chiller-2 vibration check",                "hvac",       (8, 2), 30, (0, 480)),
    ("WO-7399", "floor-12 corridor lights out",             "electrical", (2, 6), 25, (0, 480)),
    ("WO-7412", "kitchen drain blockage",                   "plumbing",   (5, 0), 40, (0, 120)),
    ("WO-7408", "meeting-room M-3-02 thermostat dead",      "controls",   (3, 3), 20, (60, 300)),
    ("WO-7401", "pool pump seal inspection",                "plumbing",   (9, 6), 35, (120, 480)),
    ("WO-7409", "lobby escalator handrail sensor",          "electrical", (1, 1), 30, (0, 180)),
    ("WO-7406", "AHU-5 belt tension",                       "hvac",       (7, 7), 25, (0, 480)),
]
TECH_START = {"Somchai": (0, 0), "Priya": (4, 4), "Marco": (9, 0)}
TECH_SKILLS = {"Somchai": {"hvac", "controls"},
               "Priya":   {"electrical", "lighting", "controls"},
               "Marco":   {"plumbing", "hvac"}}


def travel_min(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Walking minutes between two campus grid points (3 min per grid step)."""
    return 3 * (abs(a[0] - b[0]) + abs(a[1] - b[1]))


# ── occupancy feed + energy-schedule what-if for the Metropolis chapter ──────────
def occupancy_feed() -> list[tuple[str, str, str, int]]:
    """(hh:mm, camera, zone, people) — a morning lobby queue + an after-hours room."""
    ev = []
    for i, n in enumerate([4, 8, 13, 19, 23, 21, 14, 6]):          # 07:50→09:00, 10-min frames
        m = 7 * 60 + 50 + i * 10
        ev.append((f"{m // 60:02d}:{m % 60:02d}", "CAM-L1", "lobby", n))
    for hhmm, n in [("21:30", 6), ("22:00", 6), ("22:30", 5)]:
        ev.append((hhmm, "CAM-M3", "meeting room M-3-02", n))
    return ev


# What the calibrated energy model (Apps 6–7 / surrogate App 8) says the schedule
# corrections are worth. Note the second one is POSITIVE — cameras also catch rooms
# the schedule wrongly calls empty; the win is a truthful model, not always fewer kWh.
SCHEDULE_WHATIF = {
    "baseline_kwh_day": 412.0,
    "corrections": [
        ("lobby AHU: flat 40 % schedule → real 08:00–09:00 peak, low after", -22.0),
        ("meeting room M-3-02: 'unoccupied after 18:00' → actually used to 22:30", +9.0),
    ],
}


# ── the standard SIM endpoint trio ───────────────────────────────────────────────
_MODELS = ["nemotron-3-nano:30b-a3b", "nemotron-3-super:120b-a12b"]
_TOK = {"nemotron-3-nano:30b-a3b": 54.0, "nemotron-3-super:120b-a12b": 20.0}


def installed_models() -> list[str]:
    return _MODELS


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40.0))


_CANNED = (
    "[staff copilot — grounded in the twin] The west ballroom is warm because the damper on "
    "VAV-1-12 — the terminal unit serving that zone, fed by AHU-2 — has been stuck at ~12 % "
    "open since about 06:00: zone temperature climbed 23.1→26.8 °C while AHU-2 supply air "
    "stayed a healthy ~13 °C, so the fault is at the box, not the plant. CMMS history shows "
    "the same actuator (ACT-24M) was replaced 14 months ago for intermittent sticking. "
    "Recommended: dispatch Somchai (HVAC/controls, on shift until 15:00) with a spare ACT-24M; "
    "per O&M manual §4.2, lock out the 24 V control circuit at panel CP-1-B before the swap. "
    "Access: ceiling void above room B-102 via service corridor S-1 — ladder and BOH badge."
)


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)
