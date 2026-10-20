#!/usr/bin/env python3
"""PART 1 · EnergyPlus anatomy — inputs, engine, outputs  [BEGINNER]

EnergyPlus is the US DOE's open-source whole-building simulation engine (C++,
maintained by NREL, github.com/NREL/EnergyPlus, docs hosted by Big Ladder). It takes a
building description (IDF/epJSON) + a weather file (EPW) and computes every zone's heat
balance for all 8760 hours of a year. This demo dissects one guest-room model.

Run:  python demos/step01_energyplus_anatomy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

try:
    import eppy  # noqa: F401
    EPPY = True
except ImportError:
    EPPY = False

PIPELINE = """\
   IDF / epJSON  ─┐
   (the building) ├─► [ENERGYPLUS] ─► CSV/ESO · SQLite · HTML ABUPS summary
   EPW weather   ─┘   zone heat balance, 8760 h   (SQLite = best programmatic path)
"""

MINI_IDF = """\
! guest_room.idf — ONE hotel guest-room zone, loads-only (heavily trimmed but real shape)
Version, 25.1;                        ! IDF is VERSION-LOCKED — two releases/yr; use the
                                      ! transition utilities to upgrade old files.
Zone, GuestRoom_501;                  ! a thermal zone = one air node the engine balances

BuildingSurface:Detailed,             ! every wall/floor/roof is a surface object;
  Wall_West, Wall, ExtWall_Constr,    ! conduction through it is solved with CTFs
  GuestRoom_501, , Outdoors, ,        ! boundary condition: outdoor air
  SunExposed, WindExposed, 4,         ! it sees sun + wind; 4 vertices follow
  0,0,3,  0,0,0,  6,0,0,  6,0,3;      ! geometry in metres (x,y,z per vertex)

FenestrationSurface:Detailed,         ! the 4 m² west window — solar gain enters here
  Window_West, Window, DblClear,
  Wall_West, , , , , 4,
  1,0,2.5,  1,0,0.5,  5,0,0.5,  5,0,2.5;

ZoneHVAC:IdealLoadsAirSystem,         ! "magic" HVAC: meets the load perfectly, no
  GuestRoom_AC, , GR_Supply, ;        ! plant/ducts — the loads-only starting point

Output:Meter, Cooling:EnergyTransfer, Hourly;   ! what to record → CSV/ESO/SQLite
"""


def main() -> None:
    view.banner("PART 1", "EnergyPlus anatomy — inputs, engine, outputs", "BEGINNER")
    if EPPY:
        print("▣ MODE: REAL — eppy importable; you can parse/patch real IDF files.\n")
    else:
        print("▣ MODE: SIM — eppy not installed; showing the file shapes + a mock run.")
        print("  go REAL:  pip install eppy   (and install EnergyPlus from energyplus.net)\n")

    print(PIPELINE)
    print("The three inputs, plainly:")
    print("  • IDF — flat text objects, one per line-group, defined by the Energy+.idd")
    print("    dictionary. epJSON is the EXACT JSON equivalent — better for codegen; convert")
    print("    either way with the ConvertInputFormat CLI that ships with EnergyPlus.")
    print("  • EPW — 8760 hourly rows: dry-bulb, dew point, RH, pressure, global/direct/")
    print("    diffuse solar, wind, sky cover. Plus .ddy design days for sizing. Sources:")
    print("    energyplus.net/weather and climate.onebuilding.org.")
    print("  • What it simulates — zone heat balance, surface conduction (CTF), daylighting,")
    print("    infiltration/AirflowNetwork, full HVAC loops (VAV/VRF/chillers/boilers),")
    print("    Fanger PMV/PPD comfort, PV. Zone timestep 4–6/hr; HVAC adapts to ~1 min.\n")

    print("A minimal one-zone hotel model (annotated):\n")
    print(MINI_IDF)

    print("run it for real (after installing EnergyPlus + a Bangkok EPW):")
    print("  $ energyplus -w BKK.epw -d out model.idf")
    print("  $ ConvertInputFormat model.idf          # → model.epJSON, same model as JSON\n")

    print(f"Mock ABUPS end-use summary — {sim.FLOOR_AREA_M2:,} m² Bangkok hotel (annual):\n")
    total_mwh = sum(m for _, m, _ in sim.END_USES)
    print(f"  {'end use':<26}{'MWh/yr':>9}{'kWh/m²·yr':>11}{'share':>8}")
    print("  " + "─" * 62)
    for name, mwh, eui in sim.END_USES:
        share = mwh / total_mwh
        bar = "█" * max(1, round(share * 30))
        print(f"  {name:<26}{mwh:>9.0f}{eui:>11.1f}{share:>7.0%}   {bar}")
    print(f"  {'TOTAL':<26}{total_mwh:>9.0f}{total_mwh*1000/sim.FLOOR_AREA_M2:>11.1f}")
    print("\n  Reading it: Bangkok is COOLING-DOMINATED — cooling+fans+pumps ≈ half the bill.")
    print("  (Mock numbers shaped like the HTML ABUPS table; SQLite output is what code reads.)\n")

    print("Takeaway: IDF/epJSON + EPW in, 8760-hour physics out — annual runs are the norm.")
    print("Next: a model that answers in MILLISECONDS instead of minutes — build 5R1C live.")


if __name__ == "__main__":
    main()
