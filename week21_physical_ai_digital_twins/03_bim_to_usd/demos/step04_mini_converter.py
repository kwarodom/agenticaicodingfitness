#!/usr/bin/env python3
"""PART 4 · A mini IFC → USD converter, live  [ADVANCED]

Build the whole idea in ~40 lines: walk a mock-IFC hotel wing (2 storeys, spaces,
one AHU + VAVs + walls), emit a USD stage that preserves hierarchy, GlobalIds and
namespaced Psets — then LINT it with a twin-readiness checklist, pass/fail style.

REAL if `pip install usd-core` (writes a genuine USD stage in memory, no GPU),
else SIM (emits faithful USD-flavored text — same structure, same lessons).

Run:  python demos/step04_mini_converter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim   # noqa: E402
import view  # noqa: E402

try:
    from pxr import Sdf, Usd, UsdGeom  # optional: pip install usd-core
    HAVE_USD = True
except Exception:  # noqa: BLE001
    HAVE_USD = False

COBIE_GATE = 90  # % completeness required per maintained asset class


def _attrs(node) -> list[str]:
    """The converter's whole trick: GlobalId + namespaced Psets/COBie on every prim."""
    out = [("ifc:GlobalId", node["global_id"]), ("ifc:class", node["type"])]
    for pset, kv in node["psets"].items():
        out += [(f"ifc:{pset}:{k}", str(v)) for k, v in kv.items()]
    out += [(f"cobie:{k}", v) for k, v in node["cobie"].items() if v]
    return out


def convert_real() -> str:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    for path, node, _ in sim.flatten():
        prim = stage.DefinePrim(path, "Xform")
        for name, val in _attrs(node):
            prim.CreateAttribute(name, Sdf.ValueTypeNames.String, custom=True).Set(val)
    return stage.GetRootLayer().ExportToString()


def convert_sim() -> str:
    lines = ['#usda 1.0', '(', '    metersPerUnit = 1', '    upAxis = "Z"', ')', '']

    def emit(node, depth):
        pad = "    " * depth
        lines.append(f'{pad}def Xform "{node["name"]}"')
        lines.append(pad + "{")
        for name, val in _attrs(node):
            lines.append(f'{pad}    custom string {name} = "{val}"')
        for child in node["children"]:
            emit(child, depth + 1)
        lines.append(pad + "}")

    emit(sim.MOCK_IFC, 0)
    return "\n".join(lines) + "\n"


def lint() -> list[tuple[str, str, str]]:
    checks = [("PASS", "stage metadata", "metersPerUnit = 1.0 · upAxis = Z — declared by the converter")]
    got, total = sim.guid_coverage()
    checks.append(("PASS" if got == total else "FAIL", "GUID coverage",
                   f"{got}/{total} prims carry ifc:GlobalId ({round(100 * got / total)}%)"))
    for cls, (pct, count, missing) in sorted(sim.cobie_report().items()):
        detail = f"{pct}% complete across {count} asset(s)"
        if missing:
            detail += " — missing: " + ", ".join(missing)
        checks.append(("PASS" if pct >= COBIE_GATE else "FAIL", f"COBie · {cls}", detail))
    orphans = sim.orphaned_spaces()
    checks.append(("WARN" if orphans else "PASS", "orphaned spaces",
                   f"{len(orphans)} space(s) not parented to a storey: {', '.join(orphans) or '—'}"))
    return checks


def main() -> None:
    view.banner("PART 4", "A mini IFC → USD converter, live", "ADVANCED")
    if HAVE_USD:
        print("▣ MODE: REAL — usd-core installed; writing a genuine USD stage in memory.\n")
    else:
        print("▣ MODE: SIM — usd-core not installed; emitting faithful USD-flavored text.")
        print("  run it for real:  pip install usd-core   (no GPU needed)\n")

    usda = convert_real() if HAVE_USD else convert_sim()
    shown = usda.splitlines()
    head, rest = shown[:30], max(0, len(shown) - 30)
    print(f"Converted stage ({sum(1 for _ in sim.flatten())} IFC elements → prims):\n")
    print("\n".join("  " + ln for ln in head))
    if rest:
        print(f"  … ({rest} more lines — hierarchy, GlobalIds and Psets all preserved)")
    print()

    print(f"Twin-readiness lint (COBie gate: ≥{COBIE_GATE}% per maintained asset class):\n")
    fails = warns = 0
    for verdict, check, detail in lint():
        fails += verdict == "FAIL"
        warns += verdict == "WARN"
        print(f"  [{verdict}] {check:<18}{detail}")
    print()
    if fails:
        print(f"  ═ VERDICT: NOT TWIN-READY — {fails} fail · {warns} warn. The geometry is")
        print("    fine; the DATA failed the gate. Chase serials and install dates, not polygons.\n")
    else:
        print("  ═ VERDICT: twin-ready — geometry, identity and handover data all present.\n")

    print("Takeaway: a converter is judged by its lint report — hierarchy, GlobalIds,")
    print("Psets, COBie. Next: App 4 assembles this stage into the full building scene,")
    print("and App 5 binds live telemetry to these exact GlobalIds.")


if __name__ == "__main__":
    main()
