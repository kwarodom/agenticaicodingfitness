#!/usr/bin/env python3
"""PART 4 · A guardrailed week — fallback, override, postmortem  [ADVANCED]

Seven simulated days at rung ④ (supervised autonomy). Day 3: a zone sensor freezes
in-range → the stuck-sensor watchdog trips → automatic fallback to the Guideline 36
baseline → alert. Day 5: an operator override flows into episodic memory. One LLM
call writes the incident postmortem that consolidation stores as a semantic fact.

Run:  python demos/step04_guardrailed_week.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

DIAGRAM = """\
   telemetry ──► agent (episodic · semantic · procedural) ──► GUARDRAIL GATE ──► BMS
                     ▲                                        clamp · rate limit
                     │                                        watchdogs · G36 fallback
                     └── overrides & outcomes flow back (the flywheel) ◄──────────┘
"""


def main() -> None:
    view.banner("PART 4", "A guardrailed week — fallback, override, postmortem", "ADVANCED")
    view.mode_line()

    print("Every write the agent proposes passes the gate before it reaches the BMS:\n")
    print(DIAGRAM)

    ledger, eng, override_ep = sim.simulate_week()

    print("The week's ledger (one zone, 15-min control steps, Bangkok weather):\n")
    print(f"  {'day':<5}{'mode':<28}{'kWh':>7}{'comfort K·h':>13}{'events':>8}")
    print("  " + "─" * 63)
    for row in ledger:
        print(f"  {row['day']:<5}{row['mode']:<28}{row['kwh']:>7.1f}"
              f"{row['comfort_kh']:>13.2f}{len(row['events']):>8}")
    total_kwh = sum(r["kwh"] for r in ledger)
    total_kh = sum(r["comfort_kh"] for r in ledger)
    print("  " + "─" * 63)
    print(f"  {'week':<33}{total_kwh:>7.1f}{total_kh:>13.2f}")

    print("\nEvent log — what the guardrails actually did:\n")
    for row in ledger:
        for ev in row["events"]:
            print(f"  day {row['day']} · {ev}")

    day3 = ledger[2]
    print("\n  Note the insidious part of day 3: the sensor froze IN-RANGE at 24.0 °C, so")
    print("  the comfort watchdog saw a healthy zone while the real one warmed with the")
    print("  cooling valves closed. Only the stuck-sensor watchdog (ΔT = 0 for 2 h) could")
    print(f"  catch it — then G36 took over on the alternate sensor. Damage capped at "
          f"{day3['comfort_kh']:.1f} K·h.")

    print("\nDay 5's override, captured for the flywheel (Ch 3):\n")
    for k in ("when", "state", "rejected", "chosen", "note"):
        print(f"  {k:<9}: {override_ep[k]}")

    print(f"\nGuardrail gate counters for the week: {eng.counts['rate']} rate-limited writes "
          f"(routine morning/evening ramps) · {eng.counts['clamp']} clamps · "
          f"{eng.counts['trips']} watchdog trip(s).")

    print("\nThe last step of the loop: the agent writes the postmortem note that the")
    print("consolidation loop (Ch 2) will store as a semantic fact.\n")
    view.generate(
        "Write a 5-sentence incident postmortem for a building-ops log: on day 3 a zone "
        "temperature sensor froze in-range at 10:00, the stuck-sensor watchdog tripped at "
        "12:00, control fell back to the ASHRAE Guideline 36 baseline on the alternate "
        "sensor, and the duty engineer was paged. End with the one semantic fact memory "
        "should keep.", max_tokens=300, title="incident postmortem → semantic fact")

    print("\nTakeaway: a self-evolving operator is memory + flywheel + guardrails running")
    print("as one loop — learn from every day, but let nothing unlearned touch the plant.")
    print("App 12's capstone wires this operator into the full hotel twin.")


if __name__ == "__main__":
    main()
