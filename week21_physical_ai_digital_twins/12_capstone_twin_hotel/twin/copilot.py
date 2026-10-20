#!/usr/bin/env python3
"""The AGENTS (human half) — staff copilot + technician dispatch  (App 11 condensed).

A tool-calling copilot over the twin's four data planes — GRAPH (topology), TSDB
(history), CMMS (work orders), DOCS (O&M snippets) — plus a greedy + 2-opt
technician dispatcher, honestly labeled: a cuOpt stand-in with the same problem
shape (assignment + routing) at toy scale, not the real solver.
"""
from __future__ import annotations

from dataclasses import dataclass

# O&M doc snippets the copilot retrieves from (kept sovereign, on-box)
DOCS = [
    ("chiller low delta-p", "TC-900RT O&M §4.2: low chilled-water delta-P with pump speed "
     "steady usually means a fouled strainer, not refrigerant loss. Isolate, clean strainer, "
     "verify flow within 30 min. Shift load to the twin chiller meanwhile."),
    ("thermostat overrides", "Rooms exceeding 3 manual overrides/month: check thermostat "
     "calibration before dispatching HVAC — sensor drift mimics comfort complaints."),
    ("vav stuck damper", "VAV-R2: if measured flow is flat while the damper command moves, "
     "suspect a stuck actuator or a frozen flow transducer; replace transducer first."),
    ("precool vip", "VIP arrivals: pre-condition the suite to setpoint 60–90 min ahead, "
     "inside the comfort band — never below 21 °C."),
]


class CMMS:
    def __init__(self):
        self.orders: list[dict] = []

    def open(self, asset: str, priority: str, note: str) -> dict:
        wo = {"id": f"WO-{len(self.orders) + 1:03d}", "asset": asset,
              "priority": priority, "note": note, "status": "dispatched"}
        self.orders.append(wo)
        return wo


class Copilot:
    """Deterministic tool loop: each ask() plans tool calls over the four planes and
    prints them ReAct-style. The REASONING sentence itself comes from view.generate
    (REAL endpoint or SIM) via runtime.narrate — this class does the grounding."""

    def __init__(self, graph, tsdb, cmms: CMMS):
        self.graph, self.tsdb, self.cmms = graph, tsdb, cmms
        self.trace: list[str] = []

    def _t(self, tool: str, args: str, result: str) -> str:
        self.trace.append(f"  → ACT {tool}({args})")
        self.trace.append(f"  ← OBSERVE {result}")
        return result

    def query_graph(self, entity: str) -> str:
        feeds = self.graph.objects(entity, "feeds")
        fed_by = self.graph.subjects("feeds", entity)
        return self._t("query_graph", entity,
                       f"feeds={feeds or '—'} · fed_by={fed_by or '—'}")

    def query_tsdb(self, ref: str, hours: int = 24) -> dict:
        s = self.tsdb.stats(ref, hours)
        self._t("query_tsdb", f"{ref.split('//')[-1]}, {hours}h",
                f"n={s.get('n')} min={s.get('min')} mean={s.get('mean')} max={s.get('max')}")
        return s

    def search_docs(self, query: str) -> str:
        q = query.lower()
        best = max(DOCS, key=lambda kv: sum(w in (kv[0] + kv[1]).lower() for w in q.split()))
        self._t("search_docs", query, f"“{best[1][:64]}…”")
        return best[1]

    def open_work_order(self, asset: str, priority: str, note: str) -> dict:
        wo = self.cmms.open(asset, priority, note)
        self._t("open_work_order", f"{asset}, {priority}", f"{wo['id']} dispatched")
        return wo

    def flush_trace(self) -> list[str]:
        t, self.trace = self.trace, []
        return t


# ── technician dispatch — greedy + 2-opt (a cuOpt stand-in, honestly labeled) ──
TECHS = [("Somchai", "chiller", 0), ("Anong", "controls", 9), ("Lek", "hvac", 15)]


def dispatch(jobs: list[tuple[str, str, int]]) -> dict:
    """jobs: (asset, skill, floor). Greedy assignment by skill match then distance,
    then 2-opt on each tech's floor route. Returns assignments + travel before/after."""
    assign: dict[str, list] = {t[0]: [] for t in TECHS}
    for asset, skill, floor in jobs:
        best = min(TECHS, key=lambda t: (t[1] != skill, abs(t[2] - floor) + 3 * len(assign[t[0]])))
        assign[best[0]].append((asset, floor))
    routes = {}
    for name, base_floor in [(t[0], t[2]) for t in TECHS]:
        stops = assign[name]
        if not stops:
            continue
        before = _travel(base_floor, [f for _, f in stops])
        order = _two_opt(base_floor, stops)
        after = _travel(base_floor, [f for _, f in order])
        routes[name] = {"stops": order, "floors_before": before, "floors_after": after}
    return routes


def _travel(start: int, floors: list[int]) -> int:
    total, at = 0, start
    for f in floors:
        total += abs(f - at)
        at = f
    return total


def _two_opt(start: int, stops: list) -> list:
    best = stops[:]
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if _travel(start, [f for _, f in cand]) < _travel(start, [f for _, f in best]):
                    best, improved = cand, True
    return best


if __name__ == "__main__":
    print(dispatch([("CH-2 strainer", "chiller", 0), ("VAV-05-02 transducer", "hvac", 5),
                    ("TS-1203 calibration", "controls", 12), ("VAV-18-04 damper", "hvac", 18)]))
