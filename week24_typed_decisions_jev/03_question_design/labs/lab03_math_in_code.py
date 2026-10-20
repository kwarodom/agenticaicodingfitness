#!/usr/bin/env python3
"""Lab 03-3 · Keep maths, counting and dates in CODE.

Jev is a judgment model, not a calculator. TypeSafe's own "jaggedness" page
says it can struggle with numeric precision, counting and date comparison.
Python gets these right 100% of the time, for free.

We ask Jev numeric/date/counting questions that sit near the edges, then
compare with the exact answers computed in code. Then we show the right
pattern: compute the flag in code, and give Jev only the judgment.

Run: .venv/bin/python week24/03_question_design/labs/lab03_math_in_code.py
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from jevkit import ask, banner, noul, show, step, table, validate  # noqa: E402

# Zone readings near the "more than 2.0 °C above setpoint" boundary.
READINGS = [
    {"zone_c": 26.1, "setpoint_c": 24.0},   # +2.1  → yes
    {"zone_c": 25.9, "setpoint_c": 24.0},   # +1.9  → no
    {"zone_c": 26.0, "setpoint_c": 24.0},   # +2.0  → no (not MORE than 2)
    {"zone_c": 23.5, "setpoint_c": 21.4},   # +2.1  → yes
    {"zone_c": 22.9, "setpoint_c": 21.0},   # +1.9  → no
]
# Dates in mixed formats — "is the work order overdue?" (due before 2026-09-27)
TODAY = date(2026, 9, 27)
DUE = [("2026-09-26", date(2026, 9, 26)), ("27/09/2026", date(2026, 9, 27)),
       ("Oct 1, 2026", date(2026, 10, 1)), ("9/28/26", date(2026, 9, 28))]
# Counting
ALARMS = ["HIGH_TEMP", "LOW_FLOW", "HIGH_TEMP", "COMM_LOSS", "HIGH_TEMP",
          "LOW_FLOW", "HIGH_TEMP", "HIGH_TEMP", "COMM_LOSS", "HIGH_TEMP", "HIGH_TEMP"]

banner("Lab 03-3 · maths stays in code", "numbers · dates · counting — edge cases")

step(1, "ask Jev the arithmetic questions (ONE batched call)")
state = {"readings": READINGS, "today": TODAY.isoformat(), "due_dates": [d for d, _ in DUE], "alarms": ALARMS}
Q = {}
for i in range(len(READINGS)):
    Q[f"warm_{i}"] = noul(f"Is `readings[{i}].zone_c` MORE than 2.0 degrees above `readings[{i}].setpoint_c`?")
for i in range(len(DUE)):
    Q[f"overdue_{i}"] = noul(f"Is the date `due_dates[{i}]` strictly earlier than `today`?")
Q["many_high_temp"] = noul("Does `alarms` contain HIGH_TEMP more than 6 times?")
resp = ask(state, Q, quiet=True)
a = validate(resp, Q)

step(2, "compare with exact answers from Python")
rows, wrong_or_unsure = [], 0
for i, r in enumerate(READINGS):
    truth = r["zone_c"] - r["setpoint_c"] > 2.0          # ← the one line that is always right
    p = a[f"warm_{i}"]["noul"]
    ok = (p >= 0.8 and truth) or (p <= 0.2 and not truth)
    wrong_or_unsure += not ok
    rows.append([f"{r['zone_c']} vs {r['setpoint_c']} (Δ {r['zone_c'] - r['setpoint_c']:+.1f})",
                 "yes" if truth else "no", f"{p:.2f}", "✓" if ok else "✕ wrong/unsure"])
for i, (text, d) in enumerate(DUE):
    truth = d < TODAY
    p = a[f"overdue_{i}"]["noul"]
    ok = (p >= 0.8 and truth) or (p <= 0.2 and not truth)
    wrong_or_unsure += not ok
    rows.append([f"due {text} < {TODAY}", "yes" if truth else "no", f"{p:.2f}", "✓" if ok else "✕ wrong/unsure"])
count = ALARMS.count("HIGH_TEMP")
p = a["many_high_temp"]["noul"]
ok = (p >= 0.8) == (count > 6)
wrong_or_unsure += not ok
rows.append([f"HIGH_TEMP count ({count}) > 6", "yes" if count > 6 else "no", f"{p:.2f}", "✓" if ok else "✕ wrong/unsure"])
table(rows, ["question", "code says", "Jev noul", "clear & right?"])
print(f"◆ Jev: {wrong_or_unsure}/{len(rows)} answers wrong or not clear-cut (noul between 0.2 and 0.8)")
print(f"◆ Python: {len(rows)}/{len(rows)} exact, every time, $0 — and a threshold rule needs every time")

step(3, "the right pattern: code computes the facts, Jev judges the meaning")
r = READINGS[0]
facts = {
    "operator_note": "Guests in the ballroom say it feels stuffy.",
    "computed": {                                   # exact — from code, never from the model
        "deviation_c": round(r["zone_c"] - r["setpoint_c"], 2),
        "warm_deviation": r["zone_c"] - r["setpoint_c"] > 2.0,
        "high_temp_alarm_count": count,
    },
}
QJ = {"comfort_complaint": noul("Does `operator_note` report occupants feeling uncomfortable?"),
      "consistent": noul("Is `operator_note` consistent with `computed.warm_deviation` being true?")}
show(ask(facts, QJ))
print("═ Arithmetic, dates and counts: Python. Meaning and common sense: Jev.")
