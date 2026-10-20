#!/usr/bin/env python3
"""The hotel world — live state, the tools agents act through, and the SOP corpus.

This is the *real* part of the capstone: the tools genuinely mutate a hotel
state object, so the system is production-shaped — swap the SIM brain for a
Nemotron endpoint and the exact same tools drive a real building.

Continues the smart-hotel demo from earlier weeks:
  • Week 11  ex14 Hotel Ops Command Center  → "AltoTech Grand Bangkok", the dept agents
  • Week 14  lab1_hotel_mas                 → HVAC Sensor→Optimizer + durable memory
  • Week 23  nemotron tool-calling demo      → room 1203, get_room_telemetry / dispatch_work_order
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict


# ── the building ────────────────────────────────────────────────────────────
@dataclass
class Room:
    id: str
    floor: int
    temp_c: float
    setpoint_c: float
    occupied: bool
    vip: bool = False
    overrides_this_month: int = 0


@dataclass
class HotelState:
    name: str = "AltoTech Grand Bangkok"
    now: str = "2026-07-04 09:00 +07"
    rooms: dict = field(default_factory=dict)
    energy_kw: int = 420          # current whole-building draw
    target_kw: int = 380          # energy-program target
    tickets_open: int = 12
    tickets_critical: int = 2
    work_orders: list = field(default_factory=list)

    def summary(self) -> dict:
        occ = sum(r.occupied for r in self.rooms.values())
        return {"hotel": self.name, "time": self.now,
                "occupancy": f"{occ}/{len(self.rooms)} sampled rooms",
                "energy_kw": self.energy_kw, "target_kw": self.target_kw,
                "over_target_kw": max(0, self.energy_kw - self.target_kw),
                "tickets_open": self.tickets_open, "tickets_critical": self.tickets_critical}


def fresh_state() -> HotelState:
    """A small, representative slice of the building (room 1203 kept from Week 23)."""
    rooms = {
        "1203": Room("1203", 12, 26.4, 22.0, occupied=True, vip=False, overrides_this_month=4),
        "1512": Room("1512", 15, 23.1, 23.0, occupied=True, vip=True),
        "0902": Room("0902", 9, 21.0, 24.0, occupied=False),
        "1804": Room("1804", 18, 25.2, 22.0, occupied=True, vip=False),
        "lobby": Room("lobby", 1, 25.8, 24.0, occupied=True),
    }
    return HotelState(rooms=rooms)


# ── SOP corpus — the "NeMo Retriever RAG" skill retrieves from this ───────────
# (excerpted from week15/code/05_rag/sample_docs — kept sovereign, on-box)
SOP = [
    ("comfort", "Guest rooms target 23°C when occupied; the lobby is kept at 24°C by day "
                "and may drift to 26°C overnight to save energy. Function halls pre-cool to 22°C."),
    ("critical", "An OCCUPIED guest room more than 3°C above its setpoint is guest-impacting "
                 "and is triaged CRITICAL — dispatch maintenance immediately."),
    ("energy", "Baseline draw is ~10,000 kWh/day; the optimization program targets a 30% cut "
               "without dropping comfort below agreed thresholds. Trim setpoints in UNoccupied rooms first."),
    ("overrides", "A room with more than 3 manual temperature overrides in one month is flagged "
                  "for predictive maintenance — the thermostat or sensor may be miscalibrated."),
    ("vip", "VIP-occupied rooms must not have their setpoint changed autonomously; route to a human "
            "concierge approval first. Comfort takes priority over energy for VIPs."),
    ("sensors", "Room temperature sensors sample every 60s. A CO2 reading above 1,800 ppm in an empty "
                "conference room usually indicates a calibration fault, not real occupancy."),
]


def _room_key(room) -> str:
    """Normalize 'Room 1203' / ' 1203 ' / 'rm 1203' → '1203' (the Week 23 tool-calling fix)."""
    return re.sub(r"(?i)^\s*(?:room|rm)?\s*#?\s*", "", str(room or "")).strip()


# ── the tools (a real Python "harness" the agents act through) ────────────────
class Tools:
    """Every tool returns a JSON string (OpenAI tool-result shape) and, where it
    changes the world, mutates `state`. Returns from search_sop power the RAG skill."""

    def __init__(self, state: HotelState):
        self.state = state

    # schemas the REAL Nemotron endpoint receives (OpenAI function-calling format)
    SCHEMAS = [
        {"type": "function", "function": {"name": "get_room_telemetry",
            "description": "Read live HVAC telemetry for a room.",
            "parameters": {"type": "object", "properties": {"room": {"type": "string"}}, "required": ["room"]}}},
        {"type": "function", "function": {"name": "set_setpoint",
            "description": "Set a room's HVAC setpoint in °C (subject to policy).",
            "parameters": {"type": "object", "properties": {
                "room": {"type": "string"}, "setpoint_c": {"type": "number"}}, "required": ["room", "setpoint_c"]}}},
        {"type": "function", "function": {"name": "dispatch_work_order",
            "description": "Dispatch maintenance at a priority.",
            "parameters": {"type": "object", "properties": {
                "room": {"type": "string"}, "priority": {"type": "string", "enum": ["CRITICAL", "ROUTINE"]}},
                "required": ["room", "priority"]}}},
        {"type": "function", "function": {"name": "guest_profile",
            "description": "Look up whether a room is occupied and VIP status.",
            "parameters": {"type": "object", "properties": {"room": {"type": "string"}}, "required": ["room"]}}},
        {"type": "function", "function": {"name": "search_sop",
            "description": "Retrieve the relevant hotel Standard-Operating-Procedure passage (RAG skill).",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    ]

    def run(self, name: str, args: dict) -> str:
        fn = getattr(self, name, None)
        if fn is None:
            return json.dumps({"error": f"unknown tool {name}"})
        return fn(args)

    def get_room_telemetry(self, args) -> str:
        r = self.state.rooms.get(_room_key(args.get("room")))
        if not r:
            return json.dumps({})
        return json.dumps({"room": r.id, "temp_c": r.temp_c, "setpoint_c": r.setpoint_c,
                           "occupied": r.occupied, "delta_c": round(r.temp_c - r.setpoint_c, 1),
                           "overrides_this_month": r.overrides_this_month})

    def set_setpoint(self, args) -> str:
        r = self.state.rooms.get(_room_key(args.get("room")))
        if not r:
            return json.dumps({"error": "no such room"})
        old = r.setpoint_c
        r.setpoint_c = float(args.get("setpoint_c"))
        # crude energy model: raising an unoccupied room's setpoint saves ~4 kW/°C
        saved = max(0.0, (r.setpoint_c - old)) * (4 if not r.occupied else 2)
        self.state.energy_kw = max(0, round(self.state.energy_kw - saved))
        return json.dumps({"room": r.id, "old_setpoint_c": old, "new_setpoint_c": r.setpoint_c,
                           "est_kw_saved": round(saved, 1), "building_kw": self.state.energy_kw})

    def dispatch_work_order(self, args) -> str:
        room = _room_key(args.get("room"))
        prio = args.get("priority", "ROUTINE")
        wo = {"work_order": "WO-" + room, "room": room, "priority": prio, "status": "dispatched"}
        self.state.work_orders.append(wo)
        self.state.tickets_open += 1
        if prio == "CRITICAL":
            self.state.tickets_critical += 1
        return json.dumps(wo)

    def guest_profile(self, args) -> str:
        r = self.state.rooms.get(_room_key(args.get("room")))
        if not r:
            return json.dumps({})
        return json.dumps({"room": r.id, "occupied": r.occupied, "vip": r.vip})

    def search_sop(self, args) -> str:
        q = str(args.get("query", "")).lower()
        # tiny lexical retriever over the SOP corpus (stands in for NeMo Retriever)
        scored = sorted(SOP, key=lambda kv: -sum(w in (kv[0] + " " + kv[1]).lower() for w in q.split()))
        top = scored[0]
        return json.dumps({"section": top[0], "passage": top[1]})


if __name__ == "__main__":
    s = fresh_state()
    print(json.dumps(s.summary(), indent=2))
    print(Tools(s).get_room_telemetry({"room": "Room 1203"}))
