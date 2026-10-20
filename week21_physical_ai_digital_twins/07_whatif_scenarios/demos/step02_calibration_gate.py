#!/usr/bin/env python3
"""PART 2 · The calibration gate — ASHRAE Guideline 14  [INTERMEDIATE]

Before a twin's what-if answers deserve trust, the model must reproduce the REAL
building's meters. ASHRAE Guideline 14 gives the pass/fail line: two statistics
(NMBE and CV(RMSE)) against metered data. This demo computes both for an
uncalibrated and a tuned model of the Bangkok hotel and runs the gate.

Run:  python demos/step02_calibration_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

GATE = """\
   ASHRAE Guideline 14 acceptance criteria (whole-building energy):
     monthly data:  |NMBE| ≤ 5 %    CV(RMSE) ≤ 15 %
     hourly data:   |NMBE| ≤ 10 %   CV(RMSE) ≤ 30 %
   NMBE     = does the model run systematically high or low?  (bias)
   CV(RMSE) = does it track the SHAPE month to month?          (scatter)
   Related regimes: IPMVP Option D (calibrated simulation for M&V) · FEMP M&V guidelines.
"""

MON = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def gate(name: str, simulated: list[float], measured: list[float]) -> bool:
    n, cv = sim.nmbe(measured, simulated), sim.cvrmse(measured, simulated)
    ok = abs(n) <= 5.0 and cv <= 15.0
    verdict = "PASS ✓ — what-ifs credible" if ok else "FAIL ✕ — do not trust its what-ifs"
    print(f"  {name:<24} NMBE {n:+6.2f} %  (≤ ±5)   CV(RMSE) {cv:5.2f} %  (≤ 15)   [{verdict}]")
    return ok


def main() -> None:
    view.banner("PART 2", "The calibration gate — ASHRAE Guideline 14", "INTERMEDIATE")
    print("▣ MODE: REAL — pure-stdlib statistics against 12 months of mock metered kWh.")
    print("  The mock 'real hotel' has a better chiller, heavier gains, tinted glazing")
    print("  and a 23.5 °C BMS setpoint — none of which the as-designed model knows.\n")
    print(GATE)

    measured = sim.metered_monthly()
    uncal = sim.simulate(4.5, 24.0)["monthly_kwh"]                     # as-designed guesses
    tuned = sim.simulate(sim.TRUE_PARAMS["cop"], sim.TRUE_PARAMS["setpoint"],
                         gains_scale=sim.TRUE_PARAMS["gains_scale"],
                         solar_scale=sim.TRUE_PARAMS["solar_scale"])["monthly_kwh"]

    print("Monthly HVAC electricity, metered vs the two models (MWh):\n")
    print(f"  {'month':<7}{'metered':>9}{'uncalibrated':>14}{'tuned':>9}   uncalibrated error")
    print("  " + "─" * 66)
    for i, m in enumerate(MON):
        err = (uncal[i] - measured[i]) / measured[i] * 100
        bar = "█" * min(24, round(abs(err) / 1.5))
        print(f"  {m:<7}{measured[i]/1000:>9.1f}{uncal[i]/1000:>14.1f}{tuned[i]/1000:>9.1f}"
              f"   {err:+6.1f}%  {bar}")
    print()

    print("The Guideline 14 gate (monthly criteria):\n")
    ok_u = gate("uncalibrated model", uncal, measured)
    ok_t = gate("tuned model", tuned, measured)
    print()
    print("  note the failure mode: the uncalibrated model PASSES the scatter test")
    print("  (its SHAPE is right) but fails on bias — the gate requires BOTH.\n")

    if not ok_u and ok_t:
        print("What the tuning changed: COP 4.5 → 5.0, setpoint 24.0 → 23.5 °C, internal")
        print("gains ×1.12, solar ×0.90 — four physical parameters, each defensible from")
        print("evidence (chiller submeter, BMS trend, occupancy logs, glazing spec).\n")

    print("How teams actually calibrate (in rising order of automation):")
    print("  • manual / evidence-based — walk the plant room, read submeters, fix inputs")
    print("    one defensible parameter at a time (the auditable default).")
    print("  • Bayesian calibration — posterior distributions over uncertain parameters,")
    print("    so the twin's what-ifs carry error bars.")
    print("  • GA / PSO search — let an optimizer minimize CV(RMSE); fast, but beware")
    print("    unphysical parameter combos that fit the meter for the wrong reason.")
    print("  • ORNL Autotune — the published machine-learning take on the same idea.\n")

    print("An uncalibrated twin's what-if answer is an opinion, not a prediction — put a")
    print("G14 gate in the pipeline.\n")

    print("Takeaway: calibrate FIRST, sweep SECOND — the gate is two statistics and five")
    print("lines of code. Next: FMI/FMU, the standard that lets calibrated models from")
    print("different tools co-simulate in one loop.")


if __name__ == "__main__":
    main()
