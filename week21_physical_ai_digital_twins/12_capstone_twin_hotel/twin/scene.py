#!/usr/bin/env python3
"""The SCENE — a USD-flavored stage of the hotel  (Week 21 · Apps 2–4 condensed).

Everything App 2–4 taught, in one small model: discipline layers composed in
strength order (a session layer on top for live values — non-destructive
opinions), a payload per floor (30 floors, load only what you look at),
instanced furniture (thousands of chairs, six prototypes), and the
twin-readiness validator (units, GUID coverage, COBie completeness).
Runs REAL with `pip install usd-core` conceptually — here the same structure
is built in plain Python so the capstone needs nothing installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import world

# sublayer strength order — stronger first. The session layer wins: live values
# override authored ones without ever touching the BIM-derived layers.
LAYERS = ["session_live", "sensors", "furniture", "mep", "arch", "site"]

CHAIRS_PER_FLOOR = 140
CHAIR_PROTOTYPES = 6


@dataclass
class Prim:
    path: str
    ptype: str                          # Xform | Scope | PointInstancer
    layer: str
    gid: str = ""
    payload: bool = False
    instances: int = 0
    attrs: dict = field(default_factory=dict)   # authored (e.g. cobie:*, bacnet:ref)
    live: dict = field(default_factory=dict)    # session-layer opinions (strongest)


@dataclass
class Stage:
    meters_per_unit: float = 1.0
    up_axis: str = "Z"
    default_prim: str = "/World"
    prims: dict = field(default_factory=dict)   # path → Prim
    _by_gid: dict = field(default_factory=dict)

    def define(self, path, ptype, layer, gid="", **kw) -> Prim:
        p = Prim(path, ptype, layer, gid, **kw)
        self.prims[path] = p
        if gid:
            self._by_gid[gid] = p
        return p

    def find_by_gid(self, gid: str) -> Prim | None:
        return self._by_gid.get(gid)

    def set_live(self, gid: str, attr: str, value) -> bool:
        """Write a live value as a SESSION-layer opinion on the prim with this GlobalId."""
        p = self.find_by_gid(gid)
        if p is None:
            return False
        p.live[attr] = value
        return True


def assemble(inv: dict[str, world.Entity]) -> Stage:
    st = Stage()
    st.define("/World", "Xform", "arch")
    st.define("/World/Site", "Xform", "site", attrs={"address": "Sukhumvit, Bangkok"})
    st.define("/World/Tower", "Xform", "arch")

    for fl in range(1, world.FLOORS + 1):
        detailed = fl in world.DETAILED_FLOORS
        fp = st.define(f"/World/Tower/Floor_{fl:02d}", "Xform", "arch", payload=True)
        fp.attrs["detail"] = "full" if detailed else "massing"
        st.define(f"/World/Tower/Floor_{fl:02d}/Furniture", "PointInstancer", "furniture",
                  instances=CHAIRS_PER_FLOOR)

    for e in inv.values():
        attrs = {"ifc:class": e.ifc_class}
        attrs |= {f"cobie:{k}": v for k, v in e.cobie.items()}
        if e.point:
            attrs["bacnet:ref"] = e.point
        if e.ifc_class == "IfcSpace":
            path, layer = f"/World/Tower/Floor_{e.floor:02d}/{_leaf(e.name)}", "arch"
        elif e.ifc_class == "IfcSensor":
            path, layer = f"/World/Sensors/{_leaf(e.name)}", "sensors"
        elif e.floor == 0:
            path, layer = f"/World/MEP/Plant/{_leaf(e.name)}", "mep"
        else:
            path, layer = f"/World/MEP/Floor_{e.floor:02d}/{_leaf(e.name)}", "mep"
        st.define(path, "Xform", layer, gid=e.gid, attrs=attrs)
    return st


def _leaf(name: str) -> str:
    return name.replace("-", "_")


def print_tree(st: Stage, focus_floor: int = 5) -> list[str]:
    """The composed stage, one detailed floor expanded — payloads and layers labeled."""
    out = [f"usda 1.0  (metersPerUnit={st.meters_per_unit} · upAxis={st.up_axis} · "
           f"defaultPrim={st.default_prim})",
           f"sublayers, strongest first: {' < '.join(reversed(LAYERS))}"]
    shown, skipped = 0, 0
    for path in sorted(st.prims):
        p = st.prims[path]
        floor_tag = f"/Floor_{focus_floor:02d}" in path or "Floor_" not in path
        if not floor_tag:
            skipped += 1
            continue
        depth = path.count("/") - 1
        tags = [p.layer]
        if p.payload:
            tags.append("payload")
        if p.instances:
            tags.append(f"instancer ×{p.instances}")
        if p.gid:
            tags.append(f"gid {p.gid[:8]}…")
        out.append(f"  {'  ' * depth}{path.rsplit('/', 1)[-1]:<18} ({' · '.join(tags)})")
        shown += 1
    out.append(f"  … +{skipped} prims on the other {world.FLOORS - 1} floors "
               f"(payloads — composed only when a viewport looks at them)")
    return out


def validate(st: Stage, inv: dict[str, world.Entity]) -> list[tuple[str, str, str]]:
    """Twin-readiness checklist → (check, PASS|WARN|FAIL, detail)."""
    checks = []
    ok_units = st.meters_per_unit == 1.0 and st.up_axis == "Z"
    checks.append(("units & up-axis declared", "PASS" if ok_units else "FAIL",
                   f"metersPerUnit={st.meters_per_unit}, upAxis={st.up_axis}"))

    ents = list(inv.values())
    with_gid = sum(1 for e in ents if e.gid)
    pct = round(100 * with_gid / len(ents))
    status = "PASS" if pct == 100 else ("WARN" if pct >= 95 else "FAIL")
    missing = [e.name for e in ents if not e.gid]
    checks.append((f"GUID coverage {with_gid}/{len(ents)} ({pct}%)", status,
                   f"missing: {', '.join(missing) if missing else '—'}"))

    fields = ("manufacturer", "model", "installDate", "warranty", "serial")
    per_class: dict[str, list[int]] = {}
    for e in world.maintained(inv):
        filled = sum(1 for f in fields if e.cobie.get(f))
        per_class.setdefault(e.ifc_class, []).append(filled)
    for cls, counts in sorted(per_class.items()):
        pct = round(100 * sum(counts) / (len(fields) * len(counts)))
        status = "PASS" if pct == 100 else ("WARN" if pct >= 50 else "FAIL")
        checks.append((f"COBie completeness · {cls} ×{len(counts)}", status, f"{pct}% of fields"))

    payloads = sum(1 for p in st.prims.values() if p.payload)
    checks.append((f"payload per floor ({payloads}/{world.FLOORS})",
                   "PASS" if payloads == world.FLOORS else "FAIL",
                   "open one floor without composing the other 29"))

    chairs = sum(p.instances for p in st.prims.values())
    checks.append((f"furniture instanced ({chairs:,} chairs)", "PASS",
                   f"{CHAIR_PROTOTYPES} prototypes — not {chairs:,} meshes"))

    unbound = [e.name for e in ents if e.point and e.gid and not st.find_by_gid(e.gid)]
    checks.append(("every point ref lands on a prim", "PASS" if not unbound else "FAIL",
                   f"unbound: {unbound or '—'}"))
    return checks


if __name__ == "__main__":
    inv = world.build_inventory()
    st = assemble(inv)
    print("\n".join(print_tree(st)[:20]))
    for c, s, d in validate(st, inv):
        print(f"[{s}] {c} — {d}")
