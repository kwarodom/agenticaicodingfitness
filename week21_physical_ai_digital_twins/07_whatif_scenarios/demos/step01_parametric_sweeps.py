#!/usr/bin/env python3
"""PART 1 · Parametric sweeps — ask the twin 18 questions at once  [BEGINNER]

A what-if is one question ("what if COP were 5.5?"). A parametric SWEEP asks the
whole grid of questions and ranks the answers. This demo sweeps chiller COP ×
cooling setpoint × window film over the built-in RC model of a Bangkok hotel,
then shows which real tool runs sweeps at which scale (eppy → Measures → jEPlus → besos).

Run:  python demos/step01_parametric_sweeps.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

try:
    import eppy  # noqa: F401
    REAL = True
except ImportError:
    REAL = False

GRID = """\
   sweep grid:   COP ∈ {4.5, 5.5, 6.5}  ×  setpoint ∈ {23, 24, 25 °C}  ×  film ∈ {no, yes}
                 └── 3 × 3 × 2 = 18 scenarios → 18 annual runs → one ranked table
"""

TOOLS = [
    ("eppy",                "1–100 runs",     "pythonic IDF read/modify/write — a for-loop over model variants"),
    ("OpenStudio Measures", "shareable",      "a Measure = a scripted model transformation ('apply LED retrofit');"),
    ("",                    "",               "share via the BCL (bcl.nrel.gov); run: openstudio run -w workflow.osw"),
    ("jEPlus",              "1k–100k runs",   "Java batch manager — define parameters, it fans out E+ jobs"),
    ("besos",               "optimization",   "eppy + optimizers + surrogates — search the grid instead of brute force"),
]


def main() -> None:
    view.banner("PART 1", "Parametric sweeps — 18 what-ifs, ranked", "BEGINNER")
    if REAL:
        print("▣ MODE: REAL — eppy importable; the same loop below can rewrite real IDF")
        print("  files. This chapter sweeps the built-in RC hotel so it finishes in seconds.\n")
    else:
        print("▣ MODE: SIM — eppy not installed (pip install eppy). The sweep runs on the")
        print("  built-in RC model of the Bangkok hotel — the pattern is identical.\n")

    print("One question is a what-if; the grid of questions is a sweep:\n")
    print(GRID)

    results = []
    for cop in (4.5, 5.5, 6.5):
        for sp in (23.0, 24.0, 25.0):
            for film in (False, True):
                r = sim.simulate(cop, sp, film)
                results.append((cop, sp, film, r))
    results.sort(key=lambda x: x[3]["cost_thb"])

    base = next(r for c, s, f, r in results if c == 4.5 and s == 24.0 and not f)
    print("All 18 scenarios, ranked by annual electricity cost (RC hotel, Bangkok year):\n")
    print(f"  {'#':>3} {'COP':>5} {'setpt':>7} {'film':>6} {'MWh/yr':>8} {'M THB/yr':>9} "
          f"{'comfort>25.5°C':>15}   vs baseline")
    print("  " + "─" * 84)
    for i, (cop, sp, film, r) in enumerate(results, 1):
        d = (r["cost_thb"] - base["cost_thb"]) / base["cost_thb"] * 100
        tag = "◂ baseline" if (cop, sp, film) == (4.5, 24.0, False) else f"{d:+.1f}% cost"
        print(f"  {i:>3} {cop:>5.1f} {sp:>6.1f}° {'yes' if film else 'no':>6} "
              f"{r['kwh']/1000:>8.0f} {r['cost_thb']/1e6:>9.2f} {r['comfort_viol_h']:>13} h   {tag}")
    print()

    print("Reading the table:")
    print("  • COP dominates — it divides EVERY kWh; the jump 4.5 → 6.5 is a chiller")
    print("    retrofit decision you can now price in THB per year.")
    print("  • Setpoint is free but bounded by comfort — 25 °C 'saves' by giving the")
    print("    plant permission to lose on the hottest afternoons (violation hours ↑).")
    print("  • The film's cost saving is the smallest of the three levers, but it cuts")
    print("    the solar spike that saturates the chiller — violation hours drop ~60 %.")
    print("    A sweep tells you that BEFORE you buy 4,000 m² of film.\n")

    print("Which tool at which scale (the same sweep, grown up):\n")
    print(f"  {'tool':<22}{'scale':<15}what it is")
    print("  " + "─" * 84)
    for tool, scale, what in TOOLS:
        print(f"  {tool:<22}{scale:<15}{what}")
    print()
    print("run it for real:")
    print("  $ pip install eppy               # IDF read/modify/write from Python")
    print("  $ openstudio run -w workflow.osw # apply Measures from bcl.nrel.gov\n")

    print("Takeaway: a sweep turns the twin into a decision table — but every number above")
    print("came from an UNCALIBRATED model. Next: the ASHRAE Guideline 14 gate that decides")
    print("whether these answers deserve trust.")


if __name__ == "__main__":
    main()
