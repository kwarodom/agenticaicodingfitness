#!/usr/bin/env python3
"""CH 5 · The learning operator — up the autonomy ladder, safely  [ADVANCED]

The AGENTS-control layer (Apps 9–10): a policy earns autonomy one rung at a time
(offline → shadow → advisory → supervised), with every proposal passing hard
guardrails. A simulated week shows the whole life: an operator OVERRIDE becomes
a consolidated fact, an injected sensor freeze trips the watchdog into the
Guideline 36 fallback, and the LLM writes the postmortem note — the one moment
that runs on a REAL endpoint if you have one connected.

Run:  python demos/step04_learning_operator.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402
from twin import controls, runtime, telemetry  # noqa: E402

LADDER_ART = """\
   offline ──▶ shadow ──▶ advisory ──▶ supervised          (there is no rung 5:
   (sim only)  (logs)     (human      (acts inside          unsupervised does not
                           clicks)     guardrails)          exist in a building)
"""


def main() -> None:
    view.banner("CH 5", "The learning operator — the autonomy ladder", "ADVANCED")
    view.mode_line()

    rt = runtime.build(rung="shadow")
    op = rt.operator
    print("▣ THE LADDER — autonomy is EARNED, and every rung keeps the guardrails:")
    print(LADDER_ART)
    for rung in controls.LADDER:
        here = " ← we start here" if rung == op.rung else ""
        print(f"  {rung:<11} gate: {controls.GATES[rung]}{here}")
    print(f"\n  guardrails at EVERY rung: clamp {controls.BAND_C[0]}–{controls.BAND_C[1]} °C · "
          f"rate ≤ {controls.MAX_STEP_C} °C/step · watchdog after {controls.WATCHDOG_STALE} "
          f"stale polls → G36\n")

    print("▣ WEEK IN SHADOW — the policy proposes, Guideline 36 still drives")
    for day in range(1, 8):
        d = op.step(hour=7)                      # the interesting hour: the learned pre-cool
        print(f"  day {day} 07:00  proposed {d.proposed:>5.2f} °C  applied {d.applied:>5.2f} °C  "
              f"({d.note})")
    print(f"  → gate check: {op.promote()}\n")

    print("▣ WEEK IN ADVISORY — humans click apply… or don't (that's the signal)")
    for day in range(1, 8):
        rejected = day == 2                       # Tue: the night engineer knows better
        d = op.step(hour=22, human_accepts=not rejected)
        tag = "✗ OVERRIDE" if rejected else "✓"
        print(f"  day {day} 22:00  advice {d.proposed:>5.2f} °C → {tag}  {d.note}")
        if rejected:
            rt.fly.record_override("Tue 22:05", "night engineer",
                                   "kept setback at 22:40, not 22:00",
                                   "floor-5 rooms hold temp ~40 min after checkout")
    print(f"  → gate check: {op.promote()}\n")

    print("▣ CONSOLIDATION — the override was free training data (Week 18's loop)")
    rt.fly.record("Thu 03:10", "incident", "VAV-05-02 flow sensor freeze",
                  "watchdog → G36 fallback, zero comfort excursions", 0.9)
    for note in rt.fly.consolidate():
        print(f"  + {note}")
    print()

    print("▣ SUPERVISED, DAY 1 — and the building immediately tests it")
    frozen = telemetry.reading(rt.inv["VAV-05-02"].point, 30)
    for poll in range(1, 4):
        tripped = op.watchdog(frozen)
        print(f"  poll {poll}: VAV-05-02 flow = {frozen} m³/s "
              f"{'→ ⚠ WATCHDOG TRIP — frozen sensor' if tripped else '(same value)'}")
    d = op.step(hour=14)
    print(f"  14:00  {d.note} — applied {d.applied} °C, the deterministic floor holds")
    wo = rt.cmms.open("VAV-05-02 flow transducer", "ROUTINE", "frozen reading, watchdog trip")
    print(f"  {wo['id']} opened ({wo['priority']}) — per the consolidated skill, not a truck roll\n")

    runtime.narrate(
        "Write the postmortem note for a frozen VAV-05-02 flow sensor: watchdog tripped "
        "after 3 stale polls, operator fell back to Guideline 36, comfort held.",
        title="agent postmortem — one REAL reasoning moment")
    print(f"\n  {op.reset_fallback('transducer replaced under ' + wo['id'])}")

    print("\nTakeaway: the policy earns each rung with measured gates; guardrails and the")
    print("G36 floor never leave. Honest note (App 9): tuned MPC still often beats RL on")
    print("BOPTEST — the LADDER is the durable idea, whatever sits on top. Next: a full day.")


if __name__ == "__main__":
    main()
