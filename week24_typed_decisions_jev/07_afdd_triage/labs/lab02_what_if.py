#!/usr/bin/env python3
"""Lab 07-2 · What if the telemetry changes? Deterministic code wins.

Ask Jev about the OPERATOR NOTE once (the note does not change), then feed four
different telemetry situations through the same policy. One model call, four
different outcomes — because the numbers are code's job, not the model's.

Then a second call shows a classic literal-reading trap: "No smoke" vs "smoke".

Run: .venv/bin/python week24/07_afdd_triage/labs/lab02_what_if.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE.parent))
from jevkit import ask, banner, guarded, noul, show, step, table, validate  # noqa: E402

from lab01_triage_incidents import QUESTIONS, SAFETY_WORDS, policy, prepare  # noqa: E402

NOTE = "Lobby is warm. AHU fan is running. No smoke or unusual noise reported."

WHAT_IF = [  # same note, four telemetry situations
    ("fresh, 4.2 °C warm", dict(zone_c=28.2, setpoint_c=24.0, age_seconds=90, quality_ok=True)),
    ("stale (2 h old)",    dict(zone_c=28.2, setpoint_c=24.0, age_seconds=7200, quality_ok=True)),
    ("bad quality flag",   dict(zone_c=28.2, setpoint_c=24.0, age_seconds=90, quality_ok=False)),
    ("fresh, on setpoint", dict(zone_c=24.3, setpoint_c=24.0, age_seconds=90, quality_ok=True)),
]


def main() -> None:
    banner("Lab 07-2 · what-if telemetry", "one Jev call on the note · four code-computed situations")

    step(1, "ask Jev about the note ONCE (note only — the numbers come later, from code)")
    note_state = {"operator_note": NOTE, "computed": {"note_only": True}}
    resp = ask(note_state, QUESTIONS)
    a = validate(resp, QUESTIONS)
    show(resp, top=4)

    step(2, "same answers, four telemetry situations → the policy decides")
    rows = []
    for label, reading in WHAT_IF:
        st = prepare({"operator_note": NOTE, **reading})
        rec = policy(st, a)                     # reuse the ONE set of answers
        f = st["computed"]
        rows.append([label, f"{f['deviation_c']:+.1f}", f["stale"], f["bad_quality"], f["warm_deviation"],
                     rec["proposed_queue"]])
    table(rows, ["situation", "Δ°C", "stale", "bad_q", "warm", "proposed queue"])
    print("◆ 1 model call · 4 outcomes — the override is plain Python you can unit-test")
    print("⚠ the last row: Jev still said 'cooling' from the note, yet the room is on setpoint.")
    print("  A human reads both — the note and the numbers — before anyone opens a work order.")

    step(3, "a literal-reading trap: 'No smoke' vs 'smoke'")
    notes = {"no_smoke": NOTE,
             "smoke": "Lobby is warm. AHU fan is running. Smoke smell reported near the AHU."}
    qs = guarded({k: noul(f"Does `{k}` explicitly report smoke, burning, fire or another immediate "
                          "safety concern that is currently present?") for k in notes})
    safety = validate(ask(notes, qs), qs)
    for k, text in notes.items():
        hit = bool(SAFETY_WORDS.search(text))
        print(f"» {k:<9} jev safety {safety[k]['noul']:.2f} · keyword backstop {'HIT' if hit else '-'}   “{text}”")
    print(f"◆ compare STEP 1: lab 01 asks whether the note 'explicitly MENTIONS smoke…' → "
          f"{a['safety_concern']['noul']:.2f} on the SAME 'No smoke' note.")
    print("  'No smoke' does mention smoke — Jev read the question literally. Asking whether a")
    print(f"  concern is 'currently present' gives {safety['no_smoke']['noul']:.2f}. Wording is part of the program.")
    print("⚠ the keyword backstop fires on 'No smoke' too (a false alarm). That is the right way")
    print("  round: for safety, a false alarm a person dismisses beats a miss nobody sees.")
    print("═ execute: False · no_bacnet_write: True · no_setpoint_change: True")


if __name__ == "__main__":
    main()
