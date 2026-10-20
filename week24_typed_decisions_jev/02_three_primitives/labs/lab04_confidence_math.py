#!/usr/bin/env python3
"""Lab 02-4 · What `confidence` really is — recompute it yourself.

TypeSafe computes `confidence` from the answer's own probabilities. For a
CHOICE, the docs' explainer uses exactly:

      confidence = (n · peak − 1) / (n − 1)

  n = number of options, peak = the highest probability.
  1.0 → all probability on one option;  0.0 → perfectly flat.

For a SCORE, TypeSafe uses a different distribution statistic (the docs do not
publish it). This lab measures both, and shows where Score differs: a split
between ADJACENT levels keeps confidence higher than mass on a FAR level.

Either way it measures how CONCENTRATED the answer is — not "the chance the
answer is correct on your data". You only learn that by evaluating (Lab 09).

Run: .venv/bin/python week24/02_three_primitives/labs/lab04_confidence_math.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from jevkit import ask, banner, choice, confidence_from, score, step, table, top2, validate  # noqa: E402

QUESTIONS = {
    "topic": choice("What is the main topic of `message`?", {
        "hvac": "Cooling, heating, ventilation or air conditioning",
        "energy": "Energy use, electricity bills, meters or savings",
        "maintenance": "Repairs, inspections or work orders for equipment",
        "other": "Anything else, or too vague to tell",
    }),
    "effort": score("How much work would answering `message` take?", [
        "A one-line answer anyone could give",
        "A specialist spends a few minutes",
        "A specialist investigation over hours or days",
    ]),
}

MESSAGES = [
    "The chiller tripped twice last night.",
    "Why did our electricity bill jump after the chiller was repaired?",
    "Can you look into it?",
]

banner("Lab 02-4 · confidence math", "confidence = (n·peak − 1)/(n − 1)")

step(1, "one batched call per message, then recompute confidence in plain Python")
rows = []
for msg in MESSAGES:
    resp = ask({"message": msg}, QUESTIONS, quiet=True)
    ans = validate(resp, QUESTIONS)
    for qid, a in ans.items():
        mine = confidence_from(a["probabilities"])      # ← the formula, in jevkit
        p1, _ = top2(a["probabilities"])
        label = a.get("choice", f"{a.get('score', 0):.2f}")
        same = abs(mine - a["confidence"]) <= 0.02
        verdict = "✓ same" if same else "differs" if a["type"] == "score" else "≠ ?"
        levels = " ".join(f"{a['probabilities'][str(i)]:.2f}"
                          for i in range(len(a["probabilities"]))) if a["type"] == "score" else "—"
        rows.append([msg, a["type"], label, f"{p1:.2f}", f"{a['confidence']:.2f}",
                     f"{mine:.2f}", verdict, levels])
table(rows, ["message", "type", "answer", "peak", "API conf", "peak formula", "match", "score p(0 1 2)"])
print("◆ choice: the peak formula reproduces the API confidence")
print("◆ score : often close, but NOT the same statistic — look at the rows marked 'differs'")

step(2, "a worked example by hand (choice formula)")
n, peak = 4, 0.70
print(f"│ 4 options, peak 0.70  →  (4×0.70 − 1)/(4 − 1) = {(n * peak - 1) / (n - 1):.2f}")
n, peak = 2, 0.70
print(f"│ 2 options, peak 0.70  →  (2×0.70 − 1)/(2 − 1) = {(n * peak - 1) / (n - 1):.2f}")
print("◆ same peak, different confidence: with more options, 0.70 is more decisive.")

step(3, "a gate in code — the illustrative one from the reference jev_lab.py")
print("│ accept if confidence ≥ .75 AND peak ≥ .80 AND margin (peak − 2nd) ≥ .20")
print("│ these numbers are UNCALIBRATED teaching values — tune them on your own labeled data")
print("\n═ Confidence tells you HOW SURE the distribution is, not WHETHER it is right.")
print("═ The probabilities are the ground truth; confidence is a convenient summary of them.")
