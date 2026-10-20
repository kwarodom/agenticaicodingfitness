#!/usr/bin/env python3
"""PART 3 · Optimize for scale  [INTERMEDIATE]

Mirrors 'Scene Optimization & Data Integration' of the NVIDIA learning path:
instancing for everything repeated, a payload per floor so you compose only
the working set, and the asset inventory export — walk the stage → CSV.
Naive vs optimized, with numbers.

Run:  python demos/step03_optimize_scale.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim   # noqa: E402
import view  # noqa: E402


def main() -> None:
    view.banner("PART 3", "Optimize for scale", "INTERMEDIATE")
    sim.usd_mode_line()

    print("Two rules make a building-scale stage affordable:")
    print("  • INSTANCING — anything repeated (guest-room furniture, luminaires, VAV")
    print("    boxes) is marked instanceable: 4,080 chairs share 3 prototype meshes")
    print("    instead of copying geometry 4,080 times. For VERY high counts (sprinkler")
    print("    heads, ceiling tiles) a PointInstancer stores just positions — one prim.")
    print("  • PAYLOADS — each floor sits behind a payload arc: it composes only when")
    print("    you LOAD it. Open the stage → floor shells; load F12 → F12's contents.\n")

    st = sim.build_grand_bangkok()          # 22 floors payloaded, F12+F14 detailed
    stats = st.stats()
    print(f"The mini hotel stage: {stats['authored']} authored prims · "
          f"{stats['payloads']} floor payloads · {stats['instanceable']} instanceable "
          f"prims in the F12 working set.\n")

    print("Payload working sets — compose only what you're working on:\n")
    st.unload_all()
    n0 = len(st.composed())
    st.load("/World/Tower/F12")
    n1 = len(st.composed())
    st.load("/World/Tower/F14")
    n2 = len(st.composed())
    print(f"  {'load set':<28}{'prims composed':>15}   of {stats['authored']} authored")
    print("  " + "─" * 58)
    for label, n in (("(nothing — shells only)", n0), ("F12", n1), ("F12 + F14", n2)):
        bar = "█" * max(1, round(n / stats["authored"] * 30))
        print(f"  {label:<28}{n:>15}   {bar}")
    print()

    print("At production scale (the full hotel, illustrative estimates):\n")
    print(f"  {'metric':<26}{'naive':>10}{'optimized':>11}   why")
    print("  " + "─" * 88)
    for metric, naive, opt, why in sim.OPTIMIZE:
        print(f"  {metric:<26}{naive:>10}{opt:>11}   {why}")
    print()

    # ── S3: asset inventory export — walk the composed stage → CSV ──
    rows = sim.inventory(st)
    csv_lines = ["class,count,with_globalid,coverage"]
    csv_lines += [f"{cls},{n},{gid},{cov}" for cls, n, gid, cov in rows]
    out_dir = Path(__file__).resolve().parents[1] / ".sandbox"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "inventory_f12_f14.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n")
    print("Asset inventory export — walk the composed stage, one CSV row per class:\n")
    for line in csv_lines:
        print("  " + line)
    print(f"\n  → written to {csv_path.relative_to(Path(__file__).resolve().parents[3])}")
    print("  This is the FM/procurement handshake: what's in the twin, how many, and")
    print("  whether each carries the GlobalId that App 5 needs to bind live data.\n")

    print("Run it for real (USD Composer / your Kit app):")
    print("  # right-click a prim ▸ Instanceable — repeated assets share one prototype")
    print("  # File ▸ Add Payload (per floor) — then Load/Unload from the stage tree")
    print("  # Script Editor: [p for p in stage.Traverse()] — the same inventory walk\n")

    print("Takeaway: instancing buys memory, payloads buy open time — that's the")
    print("difference between a stage nobody opens and one the whole team lives in.")
    print("Next: assemble the whole Grand Bangkok — layers, floors, sensors, waypoints.")


if __name__ == "__main__":
    main()
