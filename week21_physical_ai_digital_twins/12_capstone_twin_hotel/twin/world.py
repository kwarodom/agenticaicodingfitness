#!/usr/bin/env python3
"""The building inventory — AltoTech Grand Bangkok as data  (Week 21 · Apps 3–4).

The same hotel the Week 23 capstone's agent fleet runs (rooms 1203/1512/0902/1804
kept), now modeled the twin way: every entity carries an IFC-style GlobalId (the
join key everything binds on), COBie-grade handover attrs on maintained equipment,
and a BACnet-ish point reference where telemetry exists. 30 floors; six modeled in
detail — the rest arrive as per-floor payloads in scene.py.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

HOTEL = "AltoTech Grand Bangkok"
FLOORS = 30
DETAILED_FLOORS = [1, 5, 9, 12, 15, 18]          # modeled in detail; 30 total

# IFC GlobalId: 22 chars over the IFC base-64 alphabet. Deterministic per name,
# so every run (and every store keyed on it) agrees.
_IFC64 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"


def global_id(name: str) -> str:
    digest = hashlib.sha256(name.encode()).digest()
    return "".join(_IFC64[b & 63] for b in digest[:22])


@dataclass
class Entity:
    name: str
    ifc_class: str                      # IfcChiller, IfcSpace, IfcSensor, …
    floor: int                          # 0 = plant / site level
    gid: str = ""                       # IFC-style GlobalId (join key)
    cobie: dict = field(default_factory=dict)   # manufacturer/model/install/warranty
    point: str = ""                     # BACnet-ish point ref, "" if no telemetry
    feeds: list = field(default_factory=list)   # downstream entity names

    def __post_init__(self):
        if self.gid is None:            # deliberate gap for the validator to catch
            self.gid = ""
        elif not self.gid:
            self.gid = global_id(self.name)


def _cobie(mfr, model, install, warranty, serial="") -> dict:
    return {"manufacturer": mfr, "model": model, "installDate": install,
            "warranty": warranty, "serial": serial}


# room lore carried over from the Week 23 capstone (temp, setpoint, occupied, vip, overrides)
ROOM_STATE = {
    "0501": (23.2, 23.0, True,  False, 0), "0502": (22.8, 23.0, True,  False, 1),
    "0503": (24.9, 23.0, False, False, 0), "0504": (23.4, 23.0, True,  False, 0),
    "0902": (21.0, 24.0, False, False, 0),
    "1203": (26.4, 22.0, True,  False, 4),   # the Week 23 incident room — still flagged
    "1512": (23.1, 23.0, True,  True,  0),   # the VIP suite
    "1804": (25.2, 22.0, True,  False, 0),
    "lobby": (25.8, 24.0, True, False, 0),
}


def build_inventory() -> dict[str, Entity]:
    """name → Entity, with feeds edges: chiller → AHU → VAV → room."""
    inv: dict[str, Entity] = {}

    def add(e: Entity) -> Entity:
        inv[e.name] = e
        return e

    # ── central plant (floor 0) — COBie-complete: these get maintained ────────
    add(Entity("CH-1", "IfcChiller", 0, cobie=_cobie("TropiCool", "TC-900RT", "2023-02-14",
        "2028-02-14", "TC-CH-2023-071"), point="bacnet://plant/CH-1/kW",
        feeds=["AHU-1", "AHU-2", "AHU-5", "AHU-6"]))
    add(Entity("CH-2", "IfcChiller", 0, cobie=_cobie("TropiCool", "TC-900RT", "2023-02-14",
        "2028-02-14", "TC-CH-2023-072"), point="bacnet://plant/CH-2/kW",
        feeds=["AHU-3", "AHU-4"]))
    add(Entity("CHWP-1", "IfcPump", 0, cobie=_cobie("FlowServe", "FS-CHW-45", "2023-02-14",
        "2026-02-14", "FS-2023-118"), point="bacnet://plant/CHWP-1/speed_pct", feeds=["CH-1"]))
    add(Entity("CHWP-2", "IfcPump", 0, cobie=_cobie("FlowServe", "FS-CHW-45", "2023-02-14",
        "2026-02-14", "FS-2023-119"), point="bacnet://plant/CHWP-2/speed_pct", feeds=["CH-2"]))

    # ── AHUs — one per detailed floor group; AHU-3 serves floor 5 ─────────────
    ahu_floor = {"AHU-1": 1, "AHU-2": 1, "AHU-3": 5, "AHU-4": 12, "AHU-5": 15, "AHU-6": 18}
    for name, fl in ahu_floor.items():
        add(Entity(name, "IfcUnitaryEquipment", fl,
                   cobie=_cobie("TropiCool Air", "AHU-40k", "2024-11-02", "2029-11-02",
                                f"TC-{name}-2024"),
                   point=f"bacnet://{name.lower()}/supply_temp_c"))

    # ── VAV boxes + guest rooms on the detailed floors ────────────────────────
    vav_of_room = {"0501": "VAV-05-01", "0502": "VAV-05-02", "0503": "VAV-05-03",
                   "0504": "VAV-05-04", "0902": "VAV-09-02", "1203": "VAV-12-03",
                   "1512": "VAV-15-12", "1804": "VAV-18-04", "lobby": "VAV-01-01"}
    room_ahu = {"05": "AHU-3", "09": "AHU-2", "12": "AHU-4", "15": "AHU-5",
                "18": "AHU-6", "lo": "AHU-1"}
    for room, vav in vav_of_room.items():
        floor = 1 if room == "lobby" else int(room[:2])
        # a VAV — COBie deliberately incomplete (no serial/warranty): the
        # validator in scene.py should catch exactly this, like real handovers.
        add(Entity(vav, "IfcAirTerminalBox", floor,
                   cobie={"manufacturer": "TropiCool Air", "model": "VAV-R2",
                          "installDate": "2024-11-02", "warranty": "", "serial": ""},
                   point=f"bacnet://{vav.lower()}/flow_m3s", feeds=[room]))
        inv[room_ahu[room[:2]]].feeds.append(vav)
        add(Entity(room, "IfcSpace", floor))
        # each room has a temp sensor bound to the twin
        add(Entity(f"TS-{room}", "IfcSensor", floor, point=f"bacnet://rooms/{room}/temp_c"))

    # one legacy sensor was never re-tagged after a swap — its GUID is missing.
    add(Entity("TS-legacy-0902b", "IfcSensor", 9, gid=None,
               point="bacnet://rooms/0902/temp_c_b"))
    return inv


def by_class(inv: dict[str, Entity], ifc_class: str) -> list[Entity]:
    return [e for e in inv.values() if e.ifc_class == ifc_class]


def maintained(inv: dict[str, Entity]) -> list[Entity]:
    """Equipment a CMMS would carry — the COBie bar applies to these."""
    return [e for e in inv.values()
            if e.ifc_class in ("IfcChiller", "IfcPump", "IfcUnitaryEquipment",
                               "IfcAirTerminalBox")]


def points(inv: dict[str, Entity]) -> list[tuple[str, str]]:
    """(bacnet ref, GlobalId) for every telemetry-bearing entity."""
    return [(e.point, e.gid) for e in inv.values() if e.point]


if __name__ == "__main__":
    inv = build_inventory()
    print(f"{HOTEL}: {len(inv)} entities · {len(maintained(inv))} maintained · "
          f"{len(points(inv))} points")
    print("AHU-3 feeds:", inv["AHU-3"].feeds)
