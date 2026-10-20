#!/usr/bin/env python3
"""PART 3 · CityLearn — district-scale demand response  [ADVANCED]

CityLearn (UT Austin Intelligent Environments Lab) changes the question: not "run one
zone's HVAC" but "coordinate a DISTRICT against the grid". Buildings' thermal loads are
pre-simulated timeseries — agents dispatch batteries, hot/chilled-water storage, heat
pumps and (v2) EV charging. This demo runs a 3-building mini-district and flattens its
peak with battery dispatch.

Run:  python demos/step03_citylearn_district.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402


def main() -> None:
    view.banner("PART 3", "CityLearn — district-scale demand response", "ADVANCED")
    sim.lib_mode_line("citylearn", "CityLearn", "This demo's mini-district runs on built-in tables either way.")

    print("What CityLearn (v2.x) is, plainly:")
    print("  • DISTRICT scale, grid-interactive: thermal loads are PRE-SIMULATED timeseries;")
    print("    agents dispatch batteries, DHW/chilled-water storage, heat pumps, EV charging —")
    print("    not HVAC physics. (That's Sinergym's job — Ch 2.)")
    print("  • Central single-agent OR per-building multi-agent — THE MARL testbed for buildings.")
    print("  • v2 adds power-outage resilience (can your storage ride through an outage?).")
    print("  • KPIs are normalized vs a no-control baseline — <1.0 means you beat it.")
    print("  • The CityLearn Challenges (2020–2023, incl. NeurIPS) give datasets + leaderboards.\n")

    base = sim.district_baseline()
    grid, flow = sim.district_dispatch()
    peak = max(base)
    print(f"Our mini-district: {' + '.join(n for n, _ in sim.DISTRICT)} — one battery each")
    print(f"({sim.BATTERY['capacity_kwh']:.0f} kWh / {sim.BATTERY['power_kw']:.0f} kW, "
          f"η={sim.BATTERY['efficiency']:.2f}), dispatched as a pool to flatten the peak:\n")
    print(f"  {'h':>4}  {'no storage':<26}{'kW':>6} │ {'with batteries':<26}{'kW':>6}")
    print("  " + "─" * 78)
    for h in range(24):
        b1 = "█" * max(1, round(base[h] / peak * 24))
        b2 = "█" * max(1, round(grid[h] / peak * 24))
        act = "▲ charge" if flow[h] > 0.5 else ("▼ discharge" if flow[h] < -0.5 else "")
        print(f"  {h:>3}h  {b1:<26}{base[h]:>6.0f} │ {b2:<26}{grid[h]:>6.0f}  {act}")
    print()

    kb, ks = sim.district_kpis(base), sim.district_kpis(grid)
    print("District KPIs, CityLearn-style (normalized vs the no-storage baseline):\n")
    print(f"  {'KPI':<28}{'baseline':>10}{'w/ storage':>12}{'normalized':>12}")
    print("  " + "─" * 64)
    for k in kb:
        ratio = ks[k] / kb[k] if kb[k] else 1.0
        mark = " ✓" if ratio < 0.999 else ("  ⚠" if ratio > 1.001 else "")
        print(f"  {k:<28}{kb[k]:>10.1f}{ks[k]:>12.1f}{ratio:>11.2f}x{mark}")
    print()
    print("Reading it honestly: peak, ramping, load factor and cost all improve — but")
    print("consumption and CO₂ tick UP ~1%. Batteries shift energy, they don't save it;")
    print("round-trip losses are the price of flattening the peak. (Real CityLearn also")
    print("scores unmet comfort and unserved energy — our loads are fixed, so they're n/a.)\n")

    print("run it for real (the full district with RL agents):")
    print("  $ pip install CityLearn")
    print("  >>> from citylearn.citylearn import CityLearnEnv")
    print("  >>> env = CityLearnEnv(schema='citylearn_challenge_2022_phase_all')")
    print("  >>> obs = env.reset()   # one obs list per building → central or MARL control\n")

    print("Takeaway: at district scale the reward is the GRID (peak, ramping, CO₂, cost),")
    print("not one thermostat. Next: how do you prove your controller is actually better?")
    print("A fixed, referee-grade benchmark — BOPTEST.")


if __name__ == "__main__":
    main()
