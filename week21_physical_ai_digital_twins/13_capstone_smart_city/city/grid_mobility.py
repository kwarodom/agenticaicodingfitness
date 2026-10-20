#!/usr/bin/env python3
"""City SIMULATION — traffic flow + district energy, the what-if engines.

Mobility: a toy congestion model per road segment (speed ratio vs demand) good
enough to price a reroute. Grid: CityLearn-style district dispatch (App 9 Ch 4) —
buildings shed/shift flexible load and discharge storage to flatten the peak.
"""
from __future__ import annotations

from . import world

# Evening demand profile per segment (vehicles/hr, deterministic).
DEMAND = {"S-01": 4100, "S-02": 3800, "S-03": 2100, "S-04": 900, "S-05": 1700, "S-06": 5200}
CAPACITY_PER_LANE = 900   # veh/hr/lane


def segment_state(seg_id: str, demand_mult: float = 1.0) -> dict:
    seg = next(s for s in world.SEGMENTS if s["id"] == seg_id)
    cap = seg["lanes"] * CAPACITY_PER_LANE
    x = DEMAND[seg_id] * demand_mult / cap
    # BPR-flavored speed collapse: speed = free / (1 + 0.6 x^4)
    speed = seg["free"] / (1 + 0.6 * x ** 4)
    level = "FREE" if x < 0.7 else ("HEAVY" if x < 1.0 else "JAMMED")
    return {"id": seg_id, "name": seg["name"], "vc": round(x, 2),
            "speed": round(speed, 1), "level": level}


def reroute_whatif(closed: str, into: list[str]) -> list[dict]:
    """Close one segment (incident) and push its demand onto alternates."""
    diverted = DEMAND[closed]
    share = diverted / len(into)
    out = [dict(segment_state(closed), level="CLOSED", speed=0.0, vc=0.0)]
    for s in into:
        mult = (DEMAND[s] + share) / DEMAND[s]
        out.append(segment_state(s, mult))
    return out


# ── district energy (evening, MW, deterministic) ─────────────────────────────
HOURS = ["18:00", "19:00", "20:00", "21:00", "22:00", "23:00", "00:00", "01:00"]
BASE = [28.1, 31.4, 33.9, 35.2, 33.0, 30.1, 27.6, 26.0]     # district load, MW
GRID_EVENT_HOUR = "01:00"   # the night's second act: a substation trips


def dispatch_peak() -> dict:
    """CityLearn-style DR: shed flex + discharge storage across the 20:00–21:00 peak."""
    flex = sum(b["flex_mw"] for b in world.BUILDINGS)           # hospital contributes 0
    storage = sum(b["storage_mwh"] for b in world.BUILDINGS)
    shaved = []
    for h, mw in zip(HOURS, BASE):
        cut = 0.0
        if mw > 33.0:                       # DR window
            cut = min(flex, mw - 33.0) + min(1.5, storage / 4)  # shed + storage burst
        shaved.append(round(mw - cut, 1))
    return {"base": BASE, "shaved": shaved, "flex_mw": flex,
            "peak_before": max(BASE), "peak_after": max(shaved),
            "participants": [b["name"] for b in world.BUILDINGS if b["flex_mw"] > 0]}


def substation_trip(sub_id: str) -> dict:
    """Lose one substation; how much must the district island/shed to ride through?"""
    sub = next(s for s in world.SUBSTATIONS if s["id"] == sub_id)
    lost = sub["cap_mw"] * 0.4              # 40% of its feeder load must move NOW
    covered = min(lost, sum(b["flex_mw"] for b in world.BUILDINGS
                            if b["district"] == sub["district"]) +
                  sum(b["storage_mwh"] for b in world.BUILDINGS) / 2)
    return {"sub": sub_id, "district": sub["district"], "lost_mw": round(lost, 1),
            "covered_mw": round(covered, 1), "shortfall_mw": round(max(0, lost - covered), 1)}
