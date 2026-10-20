#!/usr/bin/env python3
"""PART 4 · Alfalfa — what-if behind a BMS-shaped API  [ADVANCED]

Alfalfa (NREL, github.com/NREL/alfalfa) wraps an EnergyPlus/Spawn model behind a
BMS-shaped REST API with Haystack-tagged points — a virtual building your agents
can be tested against BEFORE the real one. This demo drives an in-process
Alfalfa-like API through the what-if "pre-cool the ballroom 2 h before the peak
tariff window", prints the KPI deltas, then asks the model for the next scenarios.

Run:  python demos/step04_alfalfa_whatif.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

SHAPE = """\
   agent ──HTTP──►  Alfalfa API        list points → read → write setpoint → advance clock
                      │ (Haystack-tagged points, same verbs as a real BMS)
                      └──► EnergyPlus / Spawn model stepping in wall-or-warp time
"""

DAY = 302   # a cool-season weekday — the plant has headroom, so shifting works

# the what-if schedule: pre-cool 07–09, ride the slab through the 09:00 peak start
def whatif_sp(hour: int) -> float:
    if 7 <= hour < 9:
        return 22.5            # pre-cool: charge the slab off-peak
    if 9 <= hour < 22:
        return 25.0            # float up through the on-peak window
    return 24.0


def run_day(sp_of_hour) -> dict:
    """Drive AlfalfaLite through one day using only its four API verbs."""
    api = sim.AlfalfaLite(day=DAY, cop=4.5)
    kwh = kwh_pk = cost = peak_window = 0.0
    viol = 0
    for hour in range(24):
        api.write("ballroom-cooling-sp", sp_of_hour(hour))
        api.advance(1)                                   # the sim clock, not wall time
        p = api.read("chiller-elec-power")
        kwh += p
        cost += p * sim.tariff(DAY, hour)
        if sim.PEAK_START <= hour < sim.PEAK_END:
            kwh_pk += p
            peak_window = max(peak_window, p)
        if api.read("ballroom-zone-temp") > sim.COMFORT_C:
            viol += 1
    return {"kwh": kwh, "kwh_pk": kwh_pk, "cost": cost,
            "peak": peak_window, "viol": viol}


def main() -> None:
    view.banner("PART 4", "Alfalfa — what-if behind a BMS-shaped API", "ADVANCED")
    view.mode_line()
    print(SHAPE)

    api = sim.AlfalfaLite(day=DAY)
    print("The four verbs, live (AlfalfaLite — an in-process mimic of Alfalfa's API):\n")
    print("  GET /points →")
    for p in api.list_points():
        rw = "rw" if p["rw"] == "rw" else "r "
        print(f"    [{rw}] {p['name']:<22} {p['tags']}")
    print(f"  GET  /read/outdoor-air-temp        → {api.read('outdoor-air-temp')} °C")
    print(f"  POST /write/ballroom-cooling-sp 22.5 → ok={api.write('ballroom-cooling-sp', 22.5)}")
    api.advance(1)
    print(f"  POST /advance 1h                   → clock now {api.clock()}\n")

    print('The what-if: "pre-cool the ballroom 2 h before the peak tariff window"')
    print(f"  day {DAY} (cool-season weekday · on-peak THB {sim.TARIFF_ON}/kWh 09–22, "
          f"off-peak {sim.TARIFF_OFF})")
    print("  baseline: hold 24.0 °C all day · what-if: 22.5 °C 07–09, float to 25.0 °C 09–22\n")
    base = run_day(lambda h: 24.0)
    test = run_day(whatif_sp)

    print(f"  {'KPI':<26}{'baseline':>10}{'what-if':>10}{'delta':>12}")
    print("  " + "─" * 60)
    rows = [
        ("on-peak window peak kW", base["peak"],   test["peak"],   "kW"),
        ("on-peak electricity",    base["kwh_pk"], test["kwh_pk"], "kWh"),
        ("day electricity",        base["kwh"],    test["kwh"],    "kWh"),
        ("day cost",               base["cost"],   test["cost"],   "THB"),
        ("comfort hours >25.5 °C", base["viol"],   test["viol"],   "h"),
    ]
    for name, b, t, unit in rows:
        print(f"  {name:<26}{b:>10.1f}{t:>10.1f}{t - b:>+11.1f} {unit}")
    print()
    print("Reading it honestly: the slab charged off-peak shifts ENERGY and BAHT out of")
    print("the expensive window with zero comfort cost — but the coincident PEAK kW does")
    print("not move, because the 320 kW plant is pinned at its cap at midday in both runs.")
    print("Shaving that line needs a hardware lever (Ch 2's window film or a COP retrofit),")
    print("not a schedule. The virtual building told you that for free.\n")

    view.generate(
        "You are the what-if planner for a Bangkok hotel twin. KPI table for 'pre-cool "
        "the ballroom 2 h before the peak tariff window' vs baseline: on-peak peak kW "
        f"{base['peak']:.0f}→{test['peak']:.0f} (chiller cap-bound at midday), on-peak kWh "
        f"{base['kwh_pk']:.0f}→{test['kwh_pk']:.0f}, day kWh {base['kwh']:.0f}→{test['kwh']:.0f}, "
        f"day cost THB {base['cost']:.0f}→{test['cost']:.0f}, comfort violations "
        f"{base['viol']}→{test['viol']} h. Propose the next 3 scenarios worth testing, "
        "one line each.",
        max_tokens=300, title="what-if planner — the next 3 scenarios")

    print("\nrun it for real:")
    print("  $ git clone https://github.com/NREL/alfalfa && cd alfalfa && docker compose up")
    print("  # then POST your model, and hit the same verbs over HTTP\n")

    print("Takeaway: Alfalfa gives agents a BMS-shaped sandbox — the same API surface as")
    print("the real building, zero risk. Next: App 8 makes what-ifs answer in milliseconds")
    print("with PhysicsNeMo surrogates; App 12 wires this console into the capstone twin.")


if __name__ == "__main__":
    main()
