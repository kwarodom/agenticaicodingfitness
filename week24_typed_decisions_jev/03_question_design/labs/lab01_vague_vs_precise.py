#!/usr/bin/env python3
"""Lab 03-1 · Vague vs precise questions — Jev reads LITERALLY.

Jev answers the question you WROTE, not the one you MEANT. "Is this email
important?" means different things to different people — so the model has to
guess your meaning. Precise questions state the exact condition.

We ask BOTH styles in the SAME request (questions are independent, so they
cannot influence each other) and compare against the labels a human wrote
for the real business rule: "does this need same-day action by operations?".

Run: .venv/bin/python week24/03_question_design/labs/lab01_vague_vs_precise.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from jevkit import ask, banner, noul, step, table, validate  # noqa: E402

# Human labels for the REAL rule: needs same-day action by operations?
EMAILS = [
    ("Chiller 2 tripped and the mall is getting warm. Please send someone now.", True),
    ("Reminder: the energy report for Q3 is due next Friday.", False),
    ("Our CEO wants to meet you next month to discuss a partnership.", False),
    ("Gateway at Hotel B has been offline since 6am; no data is arriving.", True),
    ("URGENT!!! Limited-time 50% discount on LED retrofits, today only!", False),
    ("Tenant on level 4 reports water leaking from the FCU above the server rack.", True),
]

QUESTIONS = {
    # ✗ vague: "important" to whom? for what?
    "vague_important": noul("Is `email` important?"),
    # ✓ precise: the exact condition, with what counts and what does not
    "precise_same_day": noul(
        "Does `email` report a CURRENT equipment failure, outage, leak or loss of data "
        "at a site that operations must act on today?",
        true="A problem is happening now at a building or system we operate",
        false="Deadlines, meetings, sales offers, newsletters or future plans — even if they say urgent",
    ),
}

banner("Lab 03-1 · vague vs precise", "same state, two questions, one call per email")

step(1, "ask both questions about each email (6 calls)")
rows, right = [], {"vague_important": 0, "precise_same_day": 0}
for email, truth in EMAILS:
    resp = ask({"email": email}, QUESTIONS, quiet=True)
    a = validate(resp, QUESTIONS)
    v, p = a["vague_important"]["noul"], a["precise_same_day"]["noul"]
    right["vague_important"] += (v >= 0.5) == truth
    right["precise_same_day"] += (p >= 0.5) == truth
    rows.append([email, "yes" if truth else "no", f"{v:.2f}", f"{p:.2f}"])
table(rows, ["email", "label", "vague 'important?'", "precise same-day?"])

step(2, "score both styles against the human labels (threshold 0.5)")
n = len(EMAILS)
print(f"◆ vague question   : {right['vague_important']}/{n} agree with the business rule")
print(f"◆ precise question : {right['precise_same_day']}/{n} agree with the business rule")
print("\n═ 'Important' is not your rule — it is a word. Write the exact condition,")
print("  and put boundary cases (deadlines, 'URGENT' sales mail) into the criteria.")
