#!/usr/bin/env python3
"""PART 2 · Build a 5R1C reduced-order model — live  [INTERMEDIATE]

EnergyPlus takes minutes for a year. An ISO 13790-style 5R1C network — FIVE resistances,
ONE capacitance — takes milliseconds, and THAT is what MPC controllers and interactive
twins actually run. This demo builds the whole engine below, in ~60 lines of pure
Python, and simulates a hot Bangkok day hour by hour.

Run:  python demos/step02_5r1c_live.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

NETWORK = """\
   T_out ──R_ve──┐                     the 5 R's (shown as conductances H = 1/R, W/K):
   T_out ──R_w───┤ T_air ──R_is── T_s    H_ve ventilation air · H_w windows (massless)
                 │                 │     H_em envelope→mass · H_ms mass↔surface
   T_out ──R_em──┴──── T_mass ──R_ms     H_is surface↔air film
                       (C_m) ← the 1 C: the room's thermal mass (concrete, furniture)
"""

# ── the whole engine — one 30 m² guest room, ISO 13790-style ──────────────────
H_VE, H_W = 40.0, 11.0            # W/K: ventilation air path; 4 m² window at U≈2.8
H_EM = 60.0                       # W/K: opaque walls/roof feeding the mass node
H_MS, H_IS = 680.0, 465.0         # W/K: mass↔surface, surface↔air (ISO defaults · area)
C_M = 5.0e6                       # J/K: ONE capacitance ≈ 165 kJ/m²K · 30 m²
SETPOINT, AC_CAP = 24.0, 3500.0   # °C cooling setpoint; W fan-coil capacity


def rc_day(weather, steps_per_hour=6):
    """March the network through hourly weather. Returns (t_air, cool_w) per hour."""
    H_oa = H_VE + H_W                                  # outdoor → air, directly
    H_ma = 1.0 / (1.0 / H_MS + 1.0 / H_IS)             # mass → air via the surface node
    dt = 3600.0 / steps_per_hour                       # 6/hr — same as an E+ zone timestep
    t_mass, results = 27.0, []
    for rec in weather:
        t_out, cool_w = rec["t_out"], 0.0
        gains_air = 0.5 * rec["solar_w"] + rec["internal_w"]   # split solar: half to air,
        gains_mass = 0.5 * rec["solar_w"]                      # half absorbed by the mass
        t_air = t_mass
        for _ in range(steps_per_hour):
            # zone air is ~massless → its balance solves algebraically each step
            t_free = (H_oa * t_out + H_ma * t_mass + gains_air) / (H_oa + H_ma)
            if t_free > SETPOINT:                      # AC holds the setpoint if it can
                cool = min((H_oa + H_ma) * (t_free - SETPOINT), AC_CAP)
                t_air = t_free - cool / (H_oa + H_ma)
            else:
                cool, t_air = 0.0, t_free
            cool_w = max(cool_w, cool)
            # the ONE capacitance: integrate the mass temperature forward
            q = H_EM * (t_out - t_mass) + H_ma * (t_air - t_mass) + gains_mass
            t_mass += q * dt / C_M
        results.append((rec["hour"], t_out, t_air, cool_w, rec["occupied"]))
    return results


def main() -> None:
    view.banner("PART 2", "Build a 5R1C reduced-order model — live", "INTERMEDIATE")
    print("▣ MODE: REAL — pure-stdlib physics; nothing to install. That's the point:")
    print("  a reduced-order model is small enough to LIVE INSIDE the twin.\n")
    print(NETWORK)

    print("A hot Bangkok day (outdoor 27–36 °C, west sun, guest in 18:00–08:00, AC at 24 °C):\n")
    print(f"  {'hr':>4}{'out°C':>7}{'zone°C':>8}{'cool kW':>9}   cooling load")
    print("  " + "─" * 66)
    for h, t_out, t_air, cool_w, occ in rc_day(sim.bangkok_day()):
        bar = "█" * round(cool_w / 100)
        who = "◂guest" if occ else ""
        print(f"  {h:>4}{t_out:>7.1f}{t_air:>8.1f}{cool_w/1000:>9.2f}   {bar}{who}")
    print()
    print("Reading the chart: the load tracks the outdoor sine + the west-window solar peak")
    print("mid-afternoon, then the evening bump when the guest arrives. The thermal mass")
    print("(the 1 C) smooths and DELAYS the heat — that lag is what pre-cooling exploits.\n")

    t0 = time.perf_counter()
    day = sim.bangkok_day()
    rc_day([{**day[h % 24], "hour": h} for h in range(8760)])
    ms = (time.perf_counter() - t0) * 1000
    print(f"Now the punchline — a FULL YEAR (8760 h × 6 steps/h) through the same engine:")
    print(f"  ◆ 8760 hours simulated in {ms:.0f} ms — vs minutes for an EnergyPlus annual run.")
    print("  That speed is what model-predictive control and interactive twins require:")
    print("  an MPC solver evaluates thousands of candidate schedules per decision.\n")

    print("Honest scope: 5R1C trades fidelity for speed — no surfaces, no HVAC dynamics,")
    print("no daylighting. Its numbers are only credible after CALIBRATION against")
    print("EnergyPlus or metered data (the App 7 gate).\n")

    print("Takeaway: E+ is the slow source of truth; 5R1C is the fast approximation the twin")
    print("actually runs. Next: driving the real EnergyPlus from Python — pyenergyplus.")


if __name__ == "__main__":
    main()
