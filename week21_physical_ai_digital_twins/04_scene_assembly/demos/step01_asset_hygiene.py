#!/usr/bin/env python3
"""PART 1 · Project structure & asset hygiene  [BEGINNER]

Mirrors 'Getting Started' of NVIDIA's *Assembling Digital Twins With Omniverse
and OpenUSD* learning path — their factory becomes the AltoTech Grand Bangkok
hotel. A twin project starts as a FOLDER CONVENTION plus an ASSET GATE:
nothing enters assets/ until the validator passes it.

Run:  python demos/step01_asset_hygiene.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim   # noqa: E402
import view  # noqa: E402


def main() -> None:
    view.banner("PART 1", "Project structure & asset hygiene", "BEGINNER")
    sim.usd_mode_line()

    print("NVIDIA's learning path assembles a FACTORY; we assemble a HOTEL — the moves")
    print("are identical: organize the project, review asset metadata, validate before")
    print("you assemble. In USD Composer this is the asset library / Content browser plus")
    print("the Asset Validator extension; here it is printable, inspectable Python.\n")

    print("The twin project folder convention (S1 · project folder structure):\n")
    print(sim.PROJECT_TREE)
    print()

    print("SimReady-style metadata — what makes an asset ASSEMBLY-READY, not just a mesh:\n")
    print(f"  {'asset':<18}{'class':<11}{'units':>6}{'up':>4}{'phys':>6}"
          f"  {'pivot':<15}{'semantic':<20}{'tris':>9}")
    print("  " + "─" * 92)
    for a in sim.ASSETS:
        units = "m" if a["units"] == 1.0 else "cm!"
        print(f"  {a['name']:<18}{a['cls']:<11}{units:>6}{a['up']:>4}"
              f"{'yes' if a['physics'] else 'no':>6}  {a['pivot']:<15}"
              f"{a['semantic']:<20}{a['tris']:>9,}")
    print()

    print("Why each field matters:")
    print("  • units / up-axis — mixed conventions = a sofa 100x too big, or on its side.")
    print("  • pivot — 'base-center' makes snap-to-floor just work; random pivots fight you.")
    print("  • semantic label — tells tools (and App 11's vision models) WHAT a prim is.")
    print("  • physics flag — SimReady: it can collide and sit on a pad in a physics scene.")
    print("  • GlobalId — the IFC join key App 5 binds live BACnet data to. Non-negotiable.\n")

    # ── the validator gate: 5 checks × 8 assets, print the matrix + fixes ──
    per: dict[str, list[tuple[str, str, str]]] = {}
    for asset, check, verdict, fix in sim.validate():
        per.setdefault(asset, []).append((check, verdict, fix))

    print("Validator pass (S1 · asset validator troubleshooting) — the gate into assets/:\n")
    print(f"  {'asset':<18}{'units':>6}{'up':>4}{'mat':>5}{'GId':>5}{'tris':>6}   verdict")
    print("  " + "─" * 56)
    fails: list[tuple[str, str, str]] = []
    widths = (6, 4, 5, 5, 6)
    for asset, checks in per.items():
        cells = "".join(f"{('✓' if v == 'PASS' else '✗'):>{w}}"
                        for (_, v, _), w in zip(checks, widths))
        bad = [(c, f) for c, v, f in checks if v == "FAIL"]
        fails += [(asset, c, f) for c, f in bad]
        print(f"  {asset:<18}{cells}   {'FAIL' if bad else 'PASS'}")
    clean = len(per) - len({a for a, _, _ in fails})
    print(f"\n  {clean}/{len(per)} assets pass clean · fixes for the {len(fails)} failures:\n")
    for asset, check, fix in fails:
        print(f"  ✗ {asset} · {check}")
        print(f"    fix: {fix}")
    print()

    print("Run it for real (USD Composer / your Kit app):")
    print("  $ pip install usd-core            # same pxr API this repo exports to, no GPU")
    print("  # in Composer: Window ▸ Utilities ▸ Asset Validator — the same checks, one")
    print("  # click per folder; set workspace preferences ONCE (units=m, upAxis=Z) so")
    print("  # every import and every teammate agrees before assembly starts.\n")

    print("Takeaway: a twin project is a folder convention plus a gate — fix units,")
    print("up-axis, materials, GlobalIds and polygon budgets BEFORE assembly, while a fix")
    print("costs minutes. Next: bind everything from ONE central materials library and")
    print("place equipment precisely.")


if __name__ == "__main__":
    main()
