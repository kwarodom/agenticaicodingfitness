#!/usr/bin/env python3
"""The STATE — mini bus, binding service, and the three-store join  (Week 21 · App 5).

The scene mirrors the building; this makes it LIVE:
  • a mini telemetry bus (BACnet → gateway → MQTT, condensed to a generator)
  • a binding service: point ref → session-layer attr on the prim, via GlobalId
  • the three-store architecture: GRAPH (Brick-ish triples) + TSDB (in-memory
    series) + SCENE (USD stage) — all joined on the IFC GlobalId
  • the canonical spatial-temporal query: "VAVs fed by AHU-3, floor 5, last 24 h"
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from . import world


def _jitter(key: str, hour: int) -> float:
    """Deterministic per-(point,hour) noise in [-1, 1] — no RNG state, same every run."""
    h = int.from_bytes(hashlib.sha1(f"{key}:{hour}".encode()).digest()[:4], "big")
    return (h % 2000) / 1000.0 - 1.0


def reading(ref: str, hour: int) -> float:
    """The mini bus: one deterministic sample per point per hour."""
    if "temp_c" in ref:
        room = ref.split("/")[-2]
        base = world.ROOM_STATE.get(room, (23.5,))[0]
        return round(base + 0.6 * math.sin((hour - 15) / 24 * 2 * math.pi)
                     + 0.2 * _jitter(ref, hour), 1)
    if "flow_m3s" in ref:                                    # VAVs breathe with the day
        return round(max(0.05, 0.32 + 0.10 * math.sin((hour - 14) / 24 * 2 * math.pi)
                          + 0.03 * _jitter(ref, hour)), 2)
    if "/kW" in ref:                                         # chillers follow the heat
        return round(310 + 90 * math.sin((hour - 15) / 24 * 2 * math.pi)
                     + 8 * _jitter(ref, hour), 0)
    if "speed_pct" in ref:
        return round(62 + 14 * math.sin((hour - 15) / 24 * 2 * math.pi), 0)
    return 0.0


class TSDB:
    """In-memory time-series store: ref → [(hour, value)] — the Timescale/Influx seat."""

    def __init__(self):
        self.series: dict[str, list[tuple[int, float]]] = {}

    def insert(self, ref: str, hour: int, value: float) -> None:
        self.series.setdefault(ref, []).append((hour, value))

    def window(self, ref: str, hours: int = 24) -> list[float]:
        return [v for _, v in self.series.get(ref, [])[-hours:]]

    def stats(self, ref: str, hours: int = 24) -> dict:
        w = self.window(ref, hours)
        if not w:
            return {"n": 0}
        return {"n": len(w), "min": min(w), "mean": round(sum(w) / len(w), 2), "max": max(w)}


@dataclass
class Graph:
    """Brick-ish semantics: (subject, predicate, object) triples."""
    triples: list = field(default_factory=list)

    def add(self, s, p, o):
        self.triples.append((s, p, o))

    def objects(self, s, p) -> list[str]:
        return [o for s2, p2, o in self.triples if s2 == s and p2 == p]

    def subjects(self, p, o) -> list[str]:
        return [s for s, p2, o2 in self.triples if p2 == p and o2 == o]


def build_graph(inv: dict[str, world.Entity]) -> Graph:
    g = Graph()
    for e in inv.values():
        g.add(e.name, "a", e.ifc_class)
        g.add(e.name, "onFloor", str(e.floor))
        if e.point:
            g.add(e.name, "hasPoint", e.point)
        for d in e.feeds:
            g.add(e.name, "feeds", d)
    return g


class Binder:
    """The binding service: subscribes points, writes session-layer attrs by GlobalId."""

    def __init__(self, inv, stage):
        self.stage = stage
        self.table = [(ref, gid, ref.rsplit("/", 1)[-1]) for ref, gid in world.points(inv)]

    def tick(self, tsdb: TSDB, hour: int) -> tuple[int, int]:
        """One bus cycle: sample every point → TSDB + session layer. → (bound, dropped)."""
        bound = dropped = 0
        for ref, gid, attr in self.table:
            v = reading(ref, hour)
            tsdb.insert(ref, hour, v)
            if self.stage.set_live(gid, f"live:{attr}", v):
                bound += 1
            else:
                dropped += 1              # e.g. the GUID-less legacy sensor — no join key
        return bound, dropped


def backfill(binder: Binder, tsdb: TSDB, hours: int = 24) -> None:
    for h in range(hours):
        binder.tick(tsdb, h)


def spatial_temporal_query(g: Graph, tsdb: TSDB, stage, inv,
                           ahu: str = "AHU-3", floor: int = 5, hours: int = 24) -> list[dict]:
    """The canonical App 5 query, answered by joining all three stores on GlobalId:
    GRAPH walks ahu -feeds-> vav (filtered by floor), TSDB aggregates the last N hours,
    SCENE supplies the prim path a viewport would highlight."""
    rows = []
    for vav in g.objects(ahu, "feeds"):
        if str(floor) not in g.objects(vav, "onFloor"):
            continue
        e = inv[vav]
        prim = stage.find_by_gid(e.gid)
        rows.append({"vav": vav, "gid": e.gid, "room": (e.feeds or ["—"])[0],
                     "path": prim.path if prim else "(unbound)",
                     **tsdb.stats(e.point, hours)})
    return rows


if __name__ == "__main__":
    inv = world.build_inventory()
    from . import scene
    st = scene.assemble(inv)
    g, ts, b = build_graph(inv), TSDB(), Binder(inv, st)
    backfill(b, ts)
    for r in spatial_temporal_query(g, ts, st, inv):
        print(r)
