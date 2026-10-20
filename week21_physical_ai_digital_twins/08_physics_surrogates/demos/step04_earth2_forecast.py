#!/usr/bin/env python3
"""PART 4 · Earth-2 forecasts — pre-cool before the spike  [ADVANCED]

NVIDIA Earth-2 is the AI weather/climate stack: FourCastNet (global AI forecasting),
CorrDiff (generative km-scale downscaling), earth2studio (Python), Earth-2 NIMs.
For a building, forecasts are BOUNDARY CONDITIONS: feed tomorrow's hours into the
surrogate, sweep candidate schedules in milliseconds, and pre-cool before the spike.

Run:  python demos/step04_earth2_forecast.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

PRECOOL = (5, 6, 7, 8)                       # over-cool to 22 °C, off-peak power
RELIEF = (13, 14, 15, 16, 17, 18)            # drift to 25.5 °C through the spike


def main() -> None:
    view.banner("PART 4", "Earth-2 forecasts — pre-cool before the spike", "ADVANCED")
    view.mode_line()

    print("The Earth-2 stack, in one breath: FourCastNet forecasts the globe in seconds;")
    print("CorrDiff downscales generatively to km resolution; earth2studio (Python) and")
    print("Earth-2 NIMs serve it. Honest note: for ONE building, a national-weather-service")
    print("or ECMWF point-forecast API is simpler and plenty. Earth-2 earns its keep at")
    print("fleet/city scale and for extreme-event downscaling — 'this exact block, at 1 km'.\n")

    print("Tomorrow's 24 h forecast (Bangkok, heat spike to 37.8 °C mid-afternoon):\n")
    print(f"  {'hr':>4}{'out°C':>7}{'solar kW':>10}{'occ':>6}   outdoor temperature")
    print("  " + "─" * 60)
    for r in sim.forecast_24h():
        bar = "█" * round((r["t_out"] - 26.0) * 2)
        spike = " ◂ spike" if r["t_out"] > sim.ENVELOPE["t_out"][1] else ""
        print(f"  {r['hour']:>4}{r['t_out']:>7.1f}{r['solar']:>10.0f}{r['occ']:>6.0%}   {bar}{spike}")
    print()

    print("Sweep two schedules through the GUARDED surrogate (each 24 h what-if ≈ ms —")
    print("fast enough to re-run on every forecast update):")
    print("  • do nothing — hold 24.0 °C all day")
    print("  • pre-cool  — 22.0 °C at 05–08 h (off-peak), drift to 25.5 °C at 13–18 h,")
    print("    letting the slab's stored coolth carry the spike\n")
    w = sim.fit_surrogate(sim.generate_samples(200))
    base = sim.run_schedule(w, {h: 24.0 for h in range(24)})
    sp = {h: 24.0 for h in range(24)}
    sp.update({h: 22.0 for h in PRECOOL})
    sp.update({h: 25.5 for h in RELIEF})
    pre = sim.run_schedule(w, sp, precool_hours=PRECOOL, relief_hours=RELIEF)

    print("The decision table (TOU tariff: 4.3 THB/kWh on-peak 09–22 h, else 2.6):\n")
    print(f"  {'strategy':<14}{'peak kW':>9}{'kWh/day':>9}{'THB/day':>9}   notes")
    print("  " + "─" * 66)
    for name, r in (("do nothing", base), ("pre-cool", pre)):
        print(f"  {name:<14}{r['peak_kw']:>9.1f}{r['kwh']:>9.0f}{r['thb']:>9.0f}   "
              f"guard fell back to physics {r['fallbacks']} h (the spike)")
    dp = base["peak_kw"] - pre["peak_kw"]
    dthb = base["thb"] - pre["thb"]
    print(f"\n  → pre-cool shaves {dp:.1f} kW off the peak and {dthb:.0f} THB off the day")
    print("    — modest per building, but the PEAK is what sizes plants and demand charges,")
    print("    and at fleet scale this decision runs on every building, every day.\n")

    print("Note the guard doing its job: the spike hours exceed the surrogate's 36 °C")
    print("training envelope, so those hours were answered by full physics (App 8's rule).\n")

    view.generate(
        "You are the twin's advisor. Tomorrow peaks at 37.8 °C. Pre-cooling the ballroom "
        "wing to 22 °C at 05–08 h then drifting to 25.5 °C through 13–18 h lowers peak "
        "demand and the daily bill vs holding 24 °C. In 3 sentences, tell the facilities "
        "manager what to do and why.", max_tokens=300,
        title="recommendation for the facilities manager")

    print("\nTakeaway: forecast in, decision out — surrogates make the sweep instant, and")
    print("physics backstops the extremes. That's the SIMULATION layer done: App 9 gives")
    print("the AGENTS a gym to learn these controls themselves.")


if __name__ == "__main__":
    main()
