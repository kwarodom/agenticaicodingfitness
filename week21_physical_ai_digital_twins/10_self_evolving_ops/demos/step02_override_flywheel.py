#!/usr/bin/env python3
"""PART 2 · The override flywheel — learning from corrections  [INTERMEDIATE]

Every operator override is a LABELED PREFERENCE: the action the human forced (chosen)
vs the action the agent took (rejected) — exactly the data shape DPO trains on. Week
20's data flywheel (curate → customize → evaluate → promote) applied to control logs:
run it and watch corrections fall from 4 to 0 across five mornings.

Run:  python demos/step02_override_flywheel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

DIAGRAM = """\
  production control logs ──► ② CURATE ────────────► ③ CUSTOMIZE
   (① OBSERVE: actions,        keep only the           adjust the policy on the
    outcomes, overrides)       override moments        (chosen ≻ rejected) pairs
          ▲                                                     │
          └── repeat ◄── ⑤ PROMOTE ◄── ④ EVALUATE ◄─────────────┘
                          winner ships    sim gate on App 9 KPIs
                                          + LLM-judge on incident notes
"""


def main() -> None:
    view.banner("PART 2", "The override flywheel — learning from corrections", "INTERMEDIATE")
    print("▣ MODE: offline — this chapter is pure stdlib (no GPU, no endpoint, $0).\n")

    print("An override is not an annoyance to log and forget — it is the most expensive")
    print("label your building will ever produce: a domain expert, on shift, telling you")
    print("the exactly-right action for the exactly-current state. In preference form:\n")
    for k in ("state", "rejected", "chosen", "label"):
        print(f"  {k:<9}: {sim.PREF_PAIR[k]}")

    print("\nWeek 23's data flywheel (App 11 there), pointed at control logs:\n")
    print(DIAGRAM)
    for name, what in sim.FLYWHEEL_STAGES:
        print(f"  {name:<12} {what}")

    print("\nFive repeated 'morning startup' runs, one flywheel cycle between each —")
    print("Week 18's compound returns, now for a building:\n")
    rows = sim.flywheel_runs()
    print(f"  {'run':<5}{'corrections':>12}{'comfort K·h':>13}{'kWh':>7}{'operator-min':>14}")
    print("  " + "─" * 55)
    for r in rows:
        bar = "█" * (r["corrections"] * 5) or "·"
        print(f"  {r['run']:<5}{r['corrections']:>12}{r['comfort_kh']:>13.1f}"
              f"{r['kwh']:>7}{r['operator_min']:>14}   {bar}")
    first, last = rows[0], rows[-1]
    print("\n  compound returns, run 1 → run 5:")
    print(f"    • corrections     : {first['corrections']} → {last['corrections']} "
          "(the operator stops having to fix it)")
    print(f"    • comfort         : {first['comfort_kh']:.1f} → {last['comfort_kh']:.1f} K·h")
    print(f"    • energy          : {first['kwh']} → {last['kwh']} kWh "
          f"(−{round(100 * (first['kwh'] - last['kwh']) / first['kwh'])} %)")
    print(f"    • operator time   : {first['operator_min']} → {last['operator_min']} min/morning")

    print("\nThe honest fine print: the EVALUATE gate is what makes this safe to repeat —")
    print("every candidate policy must beat the incumbent on App 9's sim KPIs before it")
    print("touches the building, and an LLM-judge reads the week's incident notes for")
    print("regressions numbers can't see. No gate, no promote.")

    print("\nTakeaway: overrides are DPO-shaped gold; the flywheel spends them until run 5")
    print("needs zero corrections. But a policy that improves itself needs a ladder of")
    print("trust before it may touch a real BMS — that's next.")


if __name__ == "__main__":
    main()
