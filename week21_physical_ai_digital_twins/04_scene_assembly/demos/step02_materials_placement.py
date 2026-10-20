#!/usr/bin/env python3
"""PART 2 · Materials & precision placement  [INTERMEDIATE]

Mirrors 'Managing Given Assets' of the NVIDIA learning path: bind every
surface from ONE central materials library (change the marble once → 412
surfaces update), replace placeholder materials, place an AHU on its pad to
the centimetre, and author the CUSTOM ATTRIBUTES App 5 binds live data to.

Run:  python demos/step02_materials_placement.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim   # noqa: E402
import view  # noqa: E402

PAD = (22.0, -8.0, 40.8)          # F12 AHU pad center, from the MEP layer
DROPPED = (21.73, -7.62, 41.12)   # where drag-and-drop left the AHU, yawed 2.0°


def main() -> None:
    view.banner("PART 2", "Materials & precision placement", "INTERMEDIATE")
    sim.usd_mode_line()

    # ── S2: review/replace materials from ONE centralized library ──
    total = sum(n for _, n, _ in sim.MATERIALS)
    print("The centralized materials library (materials/ — referenced, never copied):\n")
    print(f"  {'material file':<24}{'bindings':>9}   used on")
    print("  " + "─" * 68)
    for name, n, where in sim.MATERIALS:
        print(f"  {name:<24}{n:>9,}   {where}")
    print(f"\n  {total:,} bindings resolve to {len(sim.MATERIALS)} files. Retile the marble →")
    print("  edit ONE file and 412 surfaces update everywhere. The alternative — a copy")
    print("  pasted into every asset — is thousands of files to chase, forever.\n")
    print("MDL vs UsdPreviewSurface, in one line: MDL is NVIDIA's full physically-based")
    print("material language (layered, measured); UsdPreviewSurface is the portable subset")
    print("every USD renderer reads — author MDL, keep a UsdPreviewSurface fallback.\n")

    print("Replacing the placeholders found by Ch 2's validator (rebind, don't re-import):\n")
    print(f"  {'asset':<18}{'placeholder found':<28}→ central library binding")
    print("  " + "─" * 72)
    for asset, old, new in sim.REBINDS:
        print(f"  {asset:<18}{old:<28}→ materials/{new}")
    print()

    # ── S2: precision machine placement — the AHU onto its pad ──
    dx, dy, dz = (round(p - d, 2) for p, d in zip(PAD, DROPPED))
    print("Precision placement — AHU_12_01 onto its F12 pad (drag-and-drop is not enough):\n")
    print(f"  {'step':<34}{'x':>8}{'y':>8}{'z':>8}{'yaw':>7}")
    print("  " + "─" * 66)
    print(f"  {'1 · eyeball drop':<34}"
          f"{DROPPED[0]:>8}{DROPPED[1]:>8}{DROPPED[2]:>8}{2.0:>7}")
    print(f"  {'2 · snap base to pad surface (Z)':<34}"
          f"{DROPPED[0]:>8}{DROPPED[1]:>8}{PAD[2]:>8}{2.0:>7}")
    print(f"  {'3 · snap pivot to pad center (XY)':<34}"
          f"{PAD[0]:>8}{PAD[1]:>8}{PAD[2]:>8}{2.0:>7}")
    print(f"  {'4 · zero the yaw against the wall':<34}"
          f"{PAD[0]:>8}{PAD[1]:>8}{PAD[2]:>8}{0.0:>7}")
    print(f"\n  The eyeball drop was off by Δx={dx} Δy={dy} Δz={dz} m and 2° of yaw.")
    print("  The 'base-center' pivot convention from Ch 2 is what makes steps 2–3 one")
    print("  snap each. In USD Composer: enable snapping, snap-to-face, then type exact")
    print("  values into the transform fields — 27 cm off means ducts that don't mate.\n")

    # ── S2: custom attributes — the data hooks App 5 binds to ──
    st = sim.Stage()
    st.sublayer("layers/mep.usda")
    st.define("/World", kind="assembly", layer="mep")
    ahu = st.define("/World/AHU_12_01", kind="component", layer="mep",
                    **{"alto:class": "AHU",
                       "alto:globalId": "2N1qPa0Xz5RfKq8vJ3dT12",
                       "alto:bacnetRef": "//BACnet/3012/AHU-1",
                       "alto:zone": "F12.north",
                       "alto:servicedBy": "mech-team-A",
                       "xformOp:translate": PAD})
    st.set_rel(ahu.path, "alto:serves", "/World/Tower/F12")
    print("Custom attributes on the placed prim — the twin's API surface (as .usda):\n")
    for line in st.to_usda().splitlines():
        print("  " + line)
    print("  Maintenance zone, serviced-by, BACnet reference: App 5 joins live telemetry")
    print("  onto alto:globalId and subscribes the point behind alto:bacnetRef — these")
    print("  attributes ARE the hooks that make the scene bindable.\n")

    print("Takeaway: materials are references (change once, updates everywhere),")
    print("placement is exact numbers not eyeballs, and custom attributes are the data")
    print("hooks the rest of the week hangs on. Next: make 22 floors and 4,080 chairs")
    print("affordable — instancing and payloads.")


if __name__ == "__main__":
    main()
