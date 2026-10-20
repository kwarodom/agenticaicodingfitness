#!/usr/bin/env python3
"""Live-twin binding simulator — learn STATE binding with no broker, no GPU.

Everything a living twin needs, in-process and pure stdlib:
  • Bus            — a mini pub/sub bus standing in for MQTT (or Kafka)
  • POINTS         — a BACnet/Modbus-ish point registry with the full ID join:
                     field object ↔ point ↔ equipment tag ↔ IFC GlobalId ↔ USD prim
  • TRIPLES / q()  — a tiny Brick-ish semantics graph (dict-based triples)
  • series_24h()   — a tiny in-memory TSDB (deterministic 24 h series)
  • tick_stream()  — 20 telemetry ticks: 1 Hz steady, then a 10 Hz burst
  • installed_models / tok_s / stream_generate — the standard LLM sim hooks
    (the canned answer is Ch 5's natural-language → query-plan translation)
"""
from __future__ import annotations

import math
import time

# ── mini pub/sub bus — stands in for MQTT (or Kafka) ─────────────────────────
class Bus:
    """In-process topic bus. subscribe('bldg/#', fn) matches every bldg/ topic."""

    def __init__(self):
        self._subs: list[tuple[str, object]] = []

    def subscribe(self, pattern: str, fn) -> None:
        self._subs.append((pattern, fn))

    def publish(self, topic: str, payload: dict) -> None:
        for pattern, fn in self._subs:
            if pattern == topic or (pattern.endswith("#") and topic.startswith(pattern[:-1])):
                fn(topic, payload)


# ── the point registry — the ID join the binding service performs ────────────
# field object ↔ point name ↔ equipment tag ↔ IFC GlobalId (App 3!) ↔ USD prim path
POINTS = [
    {"bus": "BACnet/IP", "obj": "device 120 · analog-input 1", "point": "AHU-3.SAT",
     "tag": "AHU-3", "unit": "°C", "kind": "supply air temp",
     "guid": "1kTvXnbbzCWw8lcMd1dR4o", "prim": "/Building/HVAC/AHU_3"},
    {"bus": "BACnet/IP", "obj": "device 231 · analog-input 5", "point": "VAV-5-01.ZNT",
     "tag": "VAV-5-01", "unit": "°C", "kind": "zone air temp",
     "guid": "2N4qP0aRb5cUdE6fGh7iJk", "prim": "/Building/Floor5/HVAC/VAV_5_01"},
    {"bus": "BACnet/IP", "obj": "device 232 · analog-input 5", "point": "VAV-5-02.ZNT",
     "tag": "VAV-5-02", "unit": "°C", "kind": "zone air temp",
     "guid": "3Wx8Yz1AbCdEfGh2iJkLmN", "prim": "/Building/Floor5/HVAC/VAV_5_02"},
    {"bus": "BACnet/IP", "obj": "device 233 · analog-input 5", "point": "VAV-5-03.ZNT",
     "tag": "VAV-5-03", "unit": "°C", "kind": "zone air temp",
     "guid": "0Qr7St2UvWxYz3AbCdEfGh", "prim": "/Building/Floor5/HVAC/VAV_5_03"},
    {"bus": "BACnet/IP", "obj": "device 234 · analog-input 5", "point": "VAV-5-04.ZNT",
     "tag": "VAV-5-04", "unit": "°C", "kind": "zone air temp",
     "guid": "1Jk9Lm3NoPqRs4TuVwXyZ0", "prim": "/Building/Floor5/HVAC/VAV_5_04"},
    {"bus": "BACnet/IP", "obj": "device 233 · analog-output 2", "point": "VAV-5-03.DMP-CMD",
     "tag": "VAV-5-03", "unit": "%", "kind": "damper command",
     "guid": "0Qr7St2UvWxYz3AbCdEfGh", "prim": "/Building/Floor5/HVAC/VAV_5_03"},
    {"bus": "BACnet/IP", "obj": "device 233 · analog-input 7", "point": "VAV-5-03.DMP-POS",
     "tag": "VAV-5-03", "unit": "%", "kind": "damper position",
     "guid": "0Qr7St2UvWxYz3AbCdEfGh", "prim": "/Building/Floor5/HVAC/VAV_5_03"},
    {"bus": "Modbus", "obj": "unit 12 · holding-reg 40021", "point": "MTR-5.kWh",
     "tag": "MTR-5", "unit": "kWh", "kind": "floor-5 submeter energy",
     "guid": "2Ab5Cd8EfGh1IjKl4MnOpQ", "prim": "/Building/Floor5/MEP/Meter_5"},
]
POINT_BY_NAME = {p["point"]: p for p in POINTS}


def topic_for(point: str) -> str:
    tag, suffix = point.split(".", 1)
    floor = "floor5" if ("-5-" in tag or tag.endswith("-5")) else "plant"
    return f"bldg/{floor}/{tag}/{suffix}"


# ── tiny Brick-ish semantics graph (dict-based triples) ──────────────────────
# Real Brick is RDF/OWL; the predicates below are the real Brick relationship names.
TRIPLES = [
    ("AHU-3", "a", "brick:AHU"), ("AHU-2", "a", "brick:AHU"),
    ("VAV-5-01", "a", "brick:VAV"), ("VAV-5-02", "a", "brick:VAV"),
    ("VAV-5-03", "a", "brick:VAV"), ("VAV-5-04", "a", "brick:VAV"),
    ("VAV-4-02", "a", "brick:VAV"), ("VAV-3-07", "a", "brick:VAV"),
    # who feeds whom (air flows this way)
    ("AHU-3", "feeds", "VAV-5-01"), ("AHU-3", "feeds", "VAV-5-02"),
    ("AHU-3", "feeds", "VAV-5-03"), ("AHU-3", "feeds", "VAV-5-04"),
    ("AHU-3", "feeds", "VAV-4-02"),          # AHU-3 also feeds a floor-4 box
    ("AHU-2", "feeds", "VAV-3-07"),          # a different AHU entirely
    # where things are
    ("VAV-5-01", "hasLocation", "Room-501"), ("VAV-5-02", "hasLocation", "Room-502"),
    ("VAV-5-03", "hasLocation", "Room-503"), ("VAV-5-04", "hasLocation", "Room-504"),
    ("VAV-4-02", "hasLocation", "Room-402"), ("VAV-3-07", "hasLocation", "Room-307"),
    ("Floor-5", "hasPart", "Room-501"), ("Floor-5", "hasPart", "Room-502"),
    ("Floor-5", "hasPart", "Room-503"), ("Floor-5", "hasPart", "Room-504"),
    ("Floor-4", "hasPart", "Room-402"), ("Floor-3", "hasPart", "Room-307"),
    # points belong to equipment
    ("VAV-5-01.ZNT", "isPointOf", "VAV-5-01"), ("VAV-5-02.ZNT", "isPointOf", "VAV-5-02"),
    ("VAV-5-03.ZNT", "isPointOf", "VAV-5-03"), ("VAV-5-04.ZNT", "isPointOf", "VAV-5-04"),
    ("VAV-5-03.DMP-CMD", "isPointOf", "VAV-5-03"),
    ("VAV-5-03.DMP-POS", "isPointOf", "VAV-5-03"),
]


def q(s=None, p=None, o=None) -> list[tuple[str, str, str]]:
    """Match triples; None = wildcard. q('AHU-3','feeds') → everything AHU-3 feeds."""
    return [t for t in TRIPLES
            if s in (None, t[0]) and p in (None, t[1]) and o in (None, t[2])]


# ── tiny in-memory TSDB — deterministic last-24 h series ─────────────────────
def series_24h(point: str) -> list[float]:
    """24 hourly zone-temp samples (oldest→newest). VAV-5-03 drifts warm — its
    damper is stuck, so the zone overheats every afternoon."""
    seed = sum(ord(c) for c in point) % 7
    out = []
    for h in range(24):
        v = 23.0 + 0.6 * math.sin((h - 6 + seed) / 24 * 2 * math.pi) + (seed - 3) * 0.1
        if point.startswith("VAV-5-03") and 10 <= h <= 19:      # stuck damper: hot PM
            v += 2.2 * math.sin((h - 10) / 9 * math.pi) + 1.0
        out.append(round(v, 1))
    return out


# ── live snapshot used by Ch 5 (heat map + alert) ────────────────────────────
LIVE_ROOMS = [  # (room, live temp °C, setpoint °C, served by)
    ("Room-501", 23.2, 23.0, "VAV-5-01"),
    ("Room-502", 22.4, 23.0, "VAV-5-02"),
    ("Room-503", 26.1, 23.0, "VAV-5-03"),
    ("Room-504", 24.2, 23.0, "VAV-5-04"),
]
DAMPER_FAULT = {"tag": "VAV-5-03", "cmd_pct": 80, "pos_pct": 12, "stuck_min": 45}


def heat_color(delta: float) -> tuple[str, tuple[float, float, float]]:
    """Deviation from setpoint → (name, displayColor RGB) for the session layer."""
    if delta <= -0.5:
        return "cool-blue", (0.25, 0.45, 0.90)
    if delta < 0.5:
        return "on-target-green", (0.30, 0.75, 0.30)
    if delta <= 2.0:
        return "warm-amber", (0.90, 0.65, 0.15)
    return "hot-red", (0.95, 0.25, 0.15)


# ── telemetry ticks for Ch 3: 10 ticks at 1 Hz, then a 10 Hz burst ───────────
def tick_stream(n: int = 20):
    """Yield (i, sim_time_s, point, value). Ticks 1–10 arrive at 1 Hz; ticks
    11–20 are a 10 Hz burst (an occupancy sensor storm / trend upload)."""
    t = 0.0
    for i in range(1, n + 1):
        val = round(22.8 + 0.4 * math.sin(i / 3.0), 2)
        yield i, round(t, 1), "VAV-5-01.ZNT", val
        t += 1.0 if i < 10 else 0.1


# ── standard LLM sim hooks (view.py calls these when no endpoint is up) ──────
_MODELS = ["nemotron-3-nano:30b-a3b", "gemma4:12b"]
_TOK = {"nemotron-3-nano:30b-a3b": 54.0, "gemma4:12b": 38.0}


def installed_models() -> list[str]:
    return _MODELS


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40.0))


_CANNED = (
    "[simulated twin copilot] QUERY PLAN — "
    "① GRAPH (Brick): AHU-3 -feeds→ ?vav · ?vav -hasLocation→ ?room · "
    "Floor-5 -hasPart→ ?room ⇒ 4 VAVs + their IFC GlobalIds. "
    "② TSDB: for each ?vav, follow isPointOf to its ZNT point and fetch the "
    "last-24 h series (one range query per point). "
    "③ USD: map GlobalId → prim path via the registry and author a highlight "
    "displayColor opinion on the SESSION layer only. "
    "Three stores, one join key — the GlobalId — and the authored asset layers "
    "stay untouched.")


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)
