#!/usr/bin/env python3
"""Scene-assembly simulator — a mini USD-ish stage + asset catalog, pure stdlib.

Models just enough of OpenUSD (prims, discipline layers, payloads, instancing,
custom attributes, relationships) to teach NVIDIA's *Assembling Digital Twins
With Omniverse and OpenUSD* learning path fully offline. In USD Composer / a
Kit app the same moves are drag-and-drop; here they are printable, inspectable
Python — and the exported .usda opens in the real thing.

Also exports the standard Week 21 endpoint hooks: installed_models(),
tok_s(model), stream_generate(prompt, model).
"""
from __future__ import annotations

import time


# ── REAL/SIM for the USD side (usd-core is optional; no GPU needed) ──────────
def have_usd() -> bool:
    try:
        from pxr import Usd  # noqa: F401
        return True
    except Exception:
        return False


def usd_mode_line() -> None:
    if have_usd():
        print("▣ MODE: REAL — usd-core importable; exported .usda files are opened "
              "with the real pxr API.")
    else:
        print("▣ MODE: SIM — built-in mini-stage (pure stdlib). Same concepts; "
              "`pip install usd-core` to verify exports with the real USD API.")
    print()


# ── the twin project folder layout (mirrors S1 of the learning path) ─────────
PROJECT_TREE = """\
grand_bangkok_twin/
├─ assets/       vendor + SimReady-style assets, one folder per class (ahu/, vav/,
│                chiller/, furniture/, luminaire/, sensor/) — referenced, never edited
├─ materials/    ONE central library (marble.mdl, carpet.mdl, …) — referenced everywhere
├─ layers/       discipline layers: site / arch / mep / furniture / sensors (.usda)
└─ config/       stage settings: metersPerUnit = 1.0 · upAxis = Z · defaultPrim = /World"""


# ── asset library with SimReady-style metadata (some deliberately dirty) ─────
# units: metersPerUnit of the asset file (1.0 = meters, 0.01 = authored in cm)
ASSETS = [
    {"name": "AHU_30k_cmh",       "cls": "AHU",       "units": 1.0,  "up": "Z",
     "material": "steel_galv",    "gid": "2N1qPa0Xz5RfKq8vJ3dTgH", "tris": 48_200,
     "pivot": "base-center", "semantic": "hvac.ahu",        "physics": True},
    {"name": "Chiller_450RT",     "cls": "Chiller",   "units": 1.0,  "up": "Y",
     "material": "steel_paint",   "gid": "0kTz9wYb14mXqCJ5rN7eLu", "tris": 96_500,
     "pivot": "base-center", "semantic": "hvac.chiller",    "physics": True},
    {"name": "VAV_reheat",        "cls": "VAV",       "units": 1.0,  "up": "Z",
     "material": "steel_galv",    "gid": "",                       "tris": 9_800,
     "pivot": "inlet-flange", "semantic": "hvac.vav",       "physics": True},
    {"name": "GuestChair_A",      "cls": "Chair",     "units": 1.0,  "up": "Z",
     "material": "fabric_wool",   "gid": "3fB2cD8eHj61sLmVxQ9pWa", "tris": 480_000,
     "pivot": "base-center", "semantic": "furniture.chair", "physics": False},
    {"name": "KingBed_typ",       "cls": "Bed",       "units": 1.0,  "up": "Z",
     "material": "",              "gid": "1aQ7rS4tUv82wXyZ0bCdEf", "tris": 62_000,
     "pivot": "base-center", "semantic": "furniture.bed",   "physics": False},
    {"name": "LobbySofa_L",       "cls": "Sofa",      "units": 0.01, "up": "Z",
     "material": "leather_tan",   "gid": "2mK5nP8qRs31tUvWxY6zAb", "tris": 88_000,
     "pivot": "base-center", "semantic": "furniture.sofa",  "physics": False},
    {"name": "CorridorLum_600",   "cls": "Luminaire", "units": 1.0,  "up": "Z",
     "material": "alu_anodized",  "gid": "0pL3mN6oQr92sTuVwX5yZc", "tris": 4_200,
     "pivot": "ceiling-mount", "semantic": "lighting.luminaire", "physics": False},
    {"name": "CO2Sensor_wall",    "cls": "Sensor",    "units": 1.0,  "up": "Z",
     "material": "abs_white",     "gid": "1cE4fG7hIj03kLmNoP8qRs", "tris": 900,
     "pivot": "wall-mount",  "semantic": "sensor.co2",      "physics": False},
]

# per-class triangle budgets (a chair does NOT need a marketing-render mesh)
TRI_BUDGET = {"AHU": 120_000, "Chiller": 150_000, "VAV": 30_000, "Chair": 40_000,
              "Bed": 80_000, "Sofa": 90_000, "Luminaire": 10_000, "Sensor": 5_000}


def validate(assets=None) -> list[tuple[str, str, str, str]]:
    """The validator pass: (asset, check, PASS/FAIL, fix). Gate before assembly."""
    rows = []
    for a in assets or ASSETS:
        checks = [
            ("units = m",   a["units"] == 1.0,
             "re-export at metersPerUnit=1.0 (this one is cm → 100x too big)"),
            ("up-axis = Z", a["up"] == "Z",
             "rotate +90° about X on import, or re-export with upAxis=Z"),
            ("material bound", bool(a["material"]),
             "bind from materials/ central library (no placeholder greys)"),
            ("GlobalId present", bool(a["gid"]),
             "carry the IFC GUID — it is the join key App 5 binds live data on"),
            ("tri budget", a["tris"] <= TRI_BUDGET.get(a["cls"], 100_000),
             f"decimate toward ≤{TRI_BUDGET.get(a['cls'], 100_000):,} tris "
             "(marketing mesh, not a twin asset)"),
        ]
        for name, ok, fix in checks:
            rows.append((a["name"], name, "PASS" if ok else "FAIL", "" if ok else fix))
    return rows


# ── central materials library (S2: manage given assets) ──────────────────────
MATERIALS = [
    ("marble_calacatta.mdl", 412, "lobby + bathroom floors/walls"),
    ("carpet_wool.mdl",      388, "guest-room + corridor floors"),
    ("fabric_wool.mdl",      4_080, "every guest/ballroom chair"),
    ("steel_galv.mdl",       468, "AHUs, VAV boxes, ducts"),
    ("glass_lowE.mdl",       1_244, "curtain wall panels"),
    ("paint_eggshell.mdl",   2_680, "interior walls/ceilings"),
]

# placeholder → central-library rebinds performed in Ch 3
REBINDS = [
    ("KingBed_typ",     "(unbound — renders grey)",  "fabric_linen.mdl"),
    ("GuestChair_A",    "chair_fabric_COPY7.mdl",    "fabric_wool.mdl"),
    ("LobbySofa_L",     "leather_tan_local.mdl",     "leather_tan.mdl"),
    ("CorridorLum_600", "alu_v2_final_FINAL.mdl",    "alu_anodized.mdl"),
]


# ── optimization before/after (S3) — building-scale estimates, illustrative ──
OPTIMIZE = [
    ("prims composed at open",  "186,400", "7,900",
     "payload floors stay unloaded; only the F12 working set composes"),
    ("unique chair meshes",     "4,080",   "3",
     "instanceable prototypes — 4,080 instances share 3 meshes"),
    ("geometry memory (est.)",  "9.2 GB",  "1.1 GB",
     "instances share prototype geometry instead of copying it"),
    ("time to open (illustr.)", "~92 s",   "~7 s",
     "payloads defer everything you didn't ask for"),
]


# ── the mini stage: prims, layers, payloads, instancing, rels ─────────────────
class Prim:
    def __init__(self, path: str, type: str = "Xform", layer: str = "arch",
                 kind: str | None = None):
        self.path, self.type, self.layer, self.kind = path, type, layer, kind
        self.attrs: dict = {}
        self.rels: dict = {}
        self.instanceable = False
        self.payload = False          # this prim is a payload arc (load on demand)

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


class Stage:
    """A tiny composed-stage model: dict of prims + layer stack + payload set."""

    def __init__(self, meters_per_unit: float = 1.0, up_axis: str = "Z"):
        self.meters_per_unit, self.up_axis = meters_per_unit, up_axis
        self.layers: list[str] = []
        self.prims: dict[str, Prim] = {}
        self._loaded: set[str] = set()     # payload paths currently loaded

    def sublayer(self, name: str) -> None:
        self.layers.append(name)

    def define(self, path: str, type: str = "Xform", layer: str = "arch",
               kind: str | None = None, instanceable: bool = False,
               payload: bool = False, **attrs) -> Prim:
        p = Prim(path, type, layer, kind)
        p.instanceable, p.payload = instanceable, payload
        p.attrs.update(attrs)
        self.prims[path] = p
        return p

    def set_rel(self, path: str, name: str, target: str) -> None:
        self.prims[path].rels[name] = target

    def load(self, *payload_paths: str) -> None:
        self._loaded.update(payload_paths)

    def unload_all(self) -> None:
        self._loaded.clear()

    def _composed(self, path: str) -> bool:
        """A prim composes iff every payload ancestor is in the load set."""
        parts = path.split("/")
        for i in range(2, len(parts)):          # ancestors only, not self
            anc = "/".join(parts[:i])
            p = self.prims.get(anc)
            if p and p.payload and anc not in self._loaded:
                return False
        return True

    def composed(self) -> list[Prim]:
        return [p for path, p in self.prims.items() if self._composed(path)]

    def stats(self) -> dict:
        comp = self.composed()
        return {"authored": len(self.prims), "composed": len(comp),
                "instanceable": sum(1 for p in comp if p.instanceable),
                "payloads": sum(1 for p in self.prims.values() if p.payload),
                "loaded": len(self._loaded)}

    # ── printable tree of the composed stage ──
    def tree(self, root: str = "/World", max_lines: int = 60) -> str:
        kids: dict[str, list[str]] = {}
        for path in self.prims:
            if not self._composed(path):
                continue
            parent = path.rsplit("/", 1)[0] or "/"
            kids.setdefault(parent, []).append(path)
        lines: list[str] = []

        def walk(path: str, indent: str) -> None:
            p = self.prims[path]
            tag = ""
            if p.payload:
                tag = "  [payload · loaded]" if path in self._loaded \
                    else "  [payload · UNLOADED]"
            if p.instanceable:
                tag += "  [instanceable]"
            lines.append(f"{indent}{p.name}  ({p.type}{' · ' + p.kind if p.kind else ''})"
                         f"{tag}   ·{p.layer}")
            for c in sorted(kids.get(path, [])):
                walk(c, indent + "  ")

        if root in self.prims:
            walk(root, "")
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"… ({len(lines) - max_lines} more prims)"]
        return "\n".join(lines)

    # ── export a real .usda (opens in USD Composer / usdview / pxr) ──
    def to_usda(self) -> str:
        kids: dict[str, list[str]] = {}
        for path in self.prims:
            parent = path.rsplit("/", 1)[0] or "/"
            kids.setdefault(parent, []).append(path)
        out = ["#usda 1.0", "(", f"    metersPerUnit = {self.meters_per_unit}",
               f'    upAxis = "{self.up_axis}"', '    defaultPrim = "World"', ")", ""]

        def emit(path: str, ind: str) -> None:
            p = self.prims[path]
            meta = []
            if p.kind:
                meta.append(f'kind = "{p.kind}"')
            if p.instanceable:
                meta.append("instanceable = true")
            head = f'{ind}def {p.type} "{p.name}"'
            out.append(head + (" (\n" + "".join(f"{ind}    {m}\n" for m in meta)
                               + f"{ind})" if meta else ""))
            out.append(f"{ind}{{")
            for k, v in p.attrs.items():
                if isinstance(v, (int, float)):
                    out.append(f"{ind}    custom double {k} = {v}")
                elif isinstance(v, (tuple, list)):
                    out.append(f"{ind}    custom double3 {k} = "
                               f"({', '.join(str(x) for x in v)})")
                else:
                    out.append(f'{ind}    custom string {k} = "{v}"')
            for k, v in p.rels.items():
                out.append(f"{ind}    custom rel {k} = <{v}>")
            for c in sorted(kids.get(path, [])):
                out.append("")
                emit(c, ind + "    ")
            out.append(f"{ind}}}")

        for top in sorted(kids.get("/", [])):
            emit(top, "")
            out.append("")
        return "\n".join(out) + "\n"


# ── assemble AltoTech Grand Bangkok (used by Ch 4 & Ch 5) ────────────────────
GUEST_FLOORS = [f"F{n:02d}" for n in range(5, 27)]        # F05..F26 → 22 floors
ROOMS_PER_FLOOR = 16

WAYPOINTS = [
    ("WP_ChillerPlant", "/World/Plant",                (-38.0, 6.0, -4.5),
     "basement central plant — 3 chillers"),
    ("WP_AHU_Room_F12", "/World/Tower/F12/MEP/AHU_12_01", (22.0, -8.0, 42.6),
     "floor-12 AHU room — filter walk-down"),
    ("WP_Lobby",        "/World/Podium/Lobby",         (0.0, 12.0, 1.5),
     "ground-floor lobby — guest experience checks"),
]


def build_grand_bangkok(load: tuple[str, ...] = ("F12",), detailed: tuple[str, ...] =
                        ("F12", "F14"), sensors: bool = True,
                        waypoints: bool = True) -> Stage:
    """A representative slice of the hotel: every floor a payload shell; the
    `detailed` floors carry MEP + instanced guest rooms. Production totals are
    in OPTIMIZE — this stage stays small enough to print."""
    st = Stage()
    for lyr in ("site", "arch", "mep", "furniture", "sensors"):
        st.sublayer(f"layers/{lyr}.usda")

    st.define("/World", kind="assembly", layer="site")
    st.define("/World/Site", kind="group", layer="site",
              **{"alto:address": "Grand Bangkok, Sukhumvit, Bangkok"})
    st.define("/World/Podium", kind="group", layer="arch")
    st.define("/World/Podium/Lobby", kind="component", layer="arch",
              **{"alto:zone": "public.lobby"})
    st.define("/World/Plant", kind="group", layer="mep")
    for i in (1, 2, 3):
        st.define(f"/World/Plant/Chiller_{i:02d}", kind="component", layer="mep",
                  **{"alto:class": "Chiller", "alto:globalId": f"0kTz9wYb14mXqCJ5rN7eL{i}",
                     "alto:bacnetRef": f"//BACnet/2001/CH-{i}"})
    st.define("/World/Tower", kind="group", layer="arch")

    # furniture prototype — ONE authored guest room, instanced everywhere
    st.define("/Prototypes", layer="furniture")
    proto = "/Prototypes/GuestRoom_KingA"
    st.define(proto, kind="component", layer="furniture", instanceable=False)
    for item in ("KingBed_typ", "GuestChair_A_1", "GuestChair_A_2", "Desk",
                 "Wardrobe", "CeilingLum"):
        st.define(f"{proto}/{item}", type="Mesh", layer="furniture")

    for f in GUEST_FLOORS:
        fp = f"/World/Tower/{f}"
        st.define(fp, kind="group", layer="arch", payload=True)
        if f not in detailed:
            continue
        z = 3.4 * int(f[1:])
        st.define(f"{fp}/Corridor", kind="component", layer="arch")
        st.define(f"{fp}/MEP", kind="group", layer="mep")
        ahu = f"{fp}/MEP/AHU_{f[1:]}_01"
        st.define(ahu, kind="component", layer="mep",
                  **{"alto:class": "AHU", "alto:globalId": f"2N1qPa0Xz5RfKq8vJ3dT{f[1:]}",
                     "alto:bacnetRef": f"//BACnet/30{f[1:]}/AHU-1",
                     "xformOp:translate": (22.0, -8.0, z)})
        st.define(f"{fp}/Rooms", kind="group", layer="arch")
        for r in range(1, ROOMS_PER_FLOOR + 1):
            room = f"{fp}/Rooms/R{f[1:]}{r:02d}"
            st.define(room, kind="component", layer="furniture", instanceable=True,
                      **{"alto:ref": proto})
            vav = f"{fp}/MEP/VAV_{f[1:]}_{r:02d}"
            st.define(vav, kind="component", layer="mep", instanceable=True,
                      **{"alto:class": "VAV",
                         "alto:globalId": f"3vAvGbKk{f[1:]}{r:02d}xQ9pWaJ3dT",
                         "alto:bacnetRef": f"//BACnet/30{f[1:]}/VAV-{r}"})
            st.set_rel(vav, "alto:serves", room)
        if sensors:
            st.define(f"{fp}/Sensors", kind="group", layer="sensors")
            for r in (1, 8, 16):
                sn = f"{fp}/Sensors/TS_{f[1:]}{r:02d}"
                st.define(sn, kind="component", layer="sensors",
                          **{"alto:class": "Sensor",
                             "alto:globalId": f"1cE4fG7hIjTS{f[1:]}{r:02d}oP8qRs",
                             "alto:pointId": f"GB-{f}-TS-{r:02d}",
                             "alto:bacnetRef": f"//BACnet/30{f[1:]}/AI-{r}"})
                st.set_rel(sn, "alto:serves", f"{fp}/MEP/VAV_{f[1:]}_{r:02d}")

    if waypoints:
        st.define("/World/Waypoints", kind="group", layer="site")
        for name, target, pos, note in WAYPOINTS:
            wp = f"/World/Waypoints/{name}"
            st.define(wp, layer="site", **{"alto:note": note, "xformOp:translate": pos})
            st.set_rel(wp, "alto:target", target)

    st.load(*(f"/World/Tower/{f}" for f in load))
    return st


def inventory(stage: Stage) -> list[tuple[str, int, int, str]]:
    """Walk the composed stage → (class, count, with-GlobalId, coverage%)."""
    by: dict[str, list[Prim]] = {}
    for p in stage.composed():
        cls = p.attrs.get("alto:class")
        if cls:
            by.setdefault(cls, []).append(p)
    rows = []
    for cls in sorted(by):
        ps = by[cls]
        gid = sum(1 for p in ps if p.attrs.get("alto:globalId"))
        rows.append((cls, len(ps), gid, f"{100 * gid // len(ps)}%"))
    return rows


# ── standard endpoint exports (canned copilot answer for the SIM path) ───────
_MODELS = ["nemotron-3-super:120b-a12b", "nemotron-3-nano:30b-a3b"]
_TOK = {"nemotron-3-super:120b-a12b": 20.0, "nemotron-3-nano:30b-a3b": 54.0}


def installed_models() -> list[str]:
    return _MODELS


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40.0))


_CANNED = ("[simulated twin copilot] Mark the 4,080 chairs instanceable so they share "
           "three prototype meshes, and put each floor behind a payload so the stage "
           "opens with only your working set composed — instancing buys memory, "
           "payloads buy open time. Keep a GlobalId on every asset: it is the join "
           "key App 5 uses to bind live BACnet data onto these exact prims.")


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)
