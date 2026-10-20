#!/usr/bin/env python3
"""CH 4 · The what-if console — calibrate, sweep, surrogate, decide  [INTERMEDIATE]

The SIMULATION layer (Apps 6–8) in one sitting: fit the 5R1C zone model to the
meter and pass the ASHRAE Guideline 14 gate (a twin's answers are only credible
AFTER calibration), sweep setpoints parametrically, fit a millisecond surrogate
with an honest training-envelope guard, then make a real operational call —
pre-cool ahead of tomorrow's forecast, priced in baht and comfort.

Run:  python demos/step03_whatif_console.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402
from twin import energy  # noqa: E402


def main() -> None:
    view.banner("CH 4", "The what-if console — calibrated, then fast", "INTERMEDIATE")
    print("▣ MODE: SIM — built-in 5R1C model (stdlib, $0). The real path: EnergyPlus +")
    print("  eppy sweeps (Apps 6–7), PhysicsNeMo surrogates (App 8). Same discipline.\n")

    sp = [23.0] * 24
    params, n, cv, ok, trail = energy.calibrate(sp)
    print("▣ STEP 1 · CALIBRATE — grid-search the zone model against the real meter")
    print(f"  {'h_tr W/K':>9}{'gains W':>9}{'CV(RMSE)%':>11}")
    print("  " + "─" * 31)
    for h_tr, gains, err in sorted(trail, key=lambda t: t[2])[:5]:
        star = " ← best" if (h_tr, gains) == (params.h_tr, params.gains_w) else ""
        print(f"  {h_tr:>9}{gains:>9}{err:>11}{star}")
    print(f"\n  ASHRAE Guideline 14 gate (hourly): NMBE {n}% (must be ≤ ±10) · "
          f"CV(RMSE) {cv}% (must be ≤ 30)")
    print(f"  → {'PASS — what-if answers are now credible' if ok else 'FAIL — do not trust this model'}\n")

    print("▣ STEP 2 · PARAMETRIC SWEEP — daily kWh per zone vs constant setpoint")
    rows = energy.sweep(params, [21.0, 22.0, 23.0, 24.0, 25.0])
    base = dict(rows)[23.0]
    for setp, kwh in rows:
        bar = "█" * round(kwh * 2.2)
        mark = "  ← today's schedule" if setp == 23.0 else f"  ({100 * (kwh - base) / base:+.0f}%)"
        print(f"  {setp:>5.1f} °C  {kwh:>6.2f} kWh  {bar}{mark}")
    print("  every +1 °C ≈ −12 % cooling energy — but comfort bounds that lever.\n")

    print("▣ STEP 3 · SURROGATE — fit once, answer in microseconds (envelope-guarded)")
    samples = []                                   # setpoint × weather-year grid
    for offs in (-1.0, 0.0, 1.0, 2.0):
        w = [round(t + offs, 1) for t in energy.BKK_WEATHER]
        for spc in (21.0, 22.0, 23.0, 24.0, 25.0):
            samples.append((spc, sum(w) / 24, sum(energy.simulate(params, [spc] * 24, w)[1])))
    sur = energy.Surrogate()
    sur.fit(samples)
    print(f"  trained on {len(samples)} calibrated-model runs "
          f"(setpoint 21–25 °C × T̄out {sur.env['t_out'][0]:.1f}–{sur.env['t_out'][1]:.1f} °C)")
    full_ms = energy.time_ms(energy.simulate, params, sp, n=50)
    sur_ms = energy.time_ms(sur.predict, 23.0, 28.5, n=2000)
    print(f"  {'engine':<26}{'per what-if':>13}{'speedup':>9}")
    print("  " + "─" * 50)
    print(f"  {'5R1C model (24 h steps)':<26}{full_ms:>10.3f} ms{'1x':>9}")
    print(f"  {'fitted surrogate':<26}{sur_ms:>10.4f} ms{full_ms / max(sur_ms, 1e-6):>8.0f}x")
    inside = sur.predict(23.0, 28.5)
    outside = sur.predict(18.0, 28.5)
    print(f"  inside envelope  predict(23.0 °C) → {inside[0]} kWh  trusted={inside[1]}")
    print(f"  OUTSIDE envelope predict(18.0 °C) → {outside[0]} kWh  trusted={outside[1]} "
          f"→ fall back to the full model")
    print("  (the honest trade Apps 8 drills: surrogates interpolate, they don't extrapolate)\n")

    hot = [round(t + 1.8, 1) for t in energy.BKK_WEATHER]
    d = energy.precool_decision(params, hot)
    print("▣ STEP 4 · DECIDE — forecast says tomorrow runs +1.8 °C hot (the Earth-2 pattern)")
    print(f"  {'strategy':<28}{'kWh/zone':>9}{'THB/zone':>9}   comfort")
    print("  " + "─" * 66)
    print(f"  {'hold 23 °C all day':<28}{d['base_kwh']:>9}{d['base_thb']:>9}   holds, pays peak tariff")
    print(f"  {'pre-cool 05–09 h (off-peak)':<28}{d['pre_kwh']:>9}{d['pre_thb']:>9}   "
          f"max excursion {d['comfort_excursion_c']} °C")
    save = d['base_thb'] - d['pre_thb']
    print(f"\n  → decision: PRE-COOL. ~{save} THB/zone/day cheaper ({100 * save / d['base_thb']:.0f}%), "
          f"comfort inside the band.")
    print("    Scaled to ~300 conditioned zones that is a real number — found in the twin,")
    print("    BEFORE touching the building.\n")

    print("run it for real:")
    print("  $ pip install eppy && energyplus -w bangkok.epw hotel.idf   # Apps 6–7")
    print("  # PhysicsNeMo for trained surrogates at CFD scale — App 8\n")

    print("Takeaway: calibrate (or your twin lies) → sweep → surrogate for speed, guarded by")
    print("its envelope → decide with money and comfort attached. Next: let an agent operate.")


if __name__ == "__main__":
    main()
