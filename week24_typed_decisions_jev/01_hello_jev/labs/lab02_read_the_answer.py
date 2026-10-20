#!/usr/bin/env python3
"""Lab 01-2 · Read a typed answer like a program.

One request, three question types (choice, noul, score) about one guest
message. Then plain Python turns the typed answers into a routing proposal.
Jev supplies judgments; YOUR code owns the decision.

Run: .venv/bin/python week24/01_hello_jev/labs/lab02_read_the_answer.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from jevkit import ask, banner, choice, noul, score, show, step, validate  # noqa: E402

STATE = {"message": "The lobby has been freezing since this morning and guests are "
                    "complaining. Please fix it before the 6pm wedding!"}

QUESTIONS = {
    "department": choice("Which team should handle `message`?", {
        "hvac": "Heating, cooling, ventilation or air-conditioning problems",
        "housekeeping": "Cleaning, towels, linen, room tidiness",
        "front_desk": "Bookings, billing, check-in and general requests",
        "unknown": "No clear request",
    }),
    "has_deadline": noul("Does `message` state an explicit deadline or time limit?"),
    "severity": score("How severe is the impact described in `message`?", [
        "No impact stated",
        "Minor inconvenience for one guest",
        "Many guests or an event affected",
        "Safety risk",
    ]),
}

banner("Lab 01-2 · read the answer like a program", "choice + noul + score in ONE request")

step(1, "ask three typed questions about one guest message")
resp = ask(STATE, QUESTIONS)
answers = validate(resp, QUESTIONS)          # fail loudly on a malformed answer
show(resp)

step(2, "your code decides — Jev only supplied the judgments")
dept = answers["department"]
deadline = answers["has_deadline"]["noul"]
severity = answers["severity"]["score"]

# A readable, testable policy. Change these numbers without touching the model.
if dept["choice"] == "unknown" or dept["confidence"] < 0.6:
    print("→ send to a human dispatcher — the department is unclear")
else:
    if severity >= 2.5:
        priority, why = "P0", "possible safety risk"
    elif severity >= 1.5 and deadline >= 0.8:
        priority, why = "P1", "event affected + explicit deadline"
    elif severity >= 1.5:
        priority, why = "P2", "several guests affected"
    else:
        priority, why = "P3", "minor, no deadline"
    print(f"→ route to {dept['choice']} queue · priority {priority} ({why})")

print("═ execute: False — this lab never pages anyone; it prints the proposal.")
print("\nTry: edit STATE['message'] above (e.g. 'Can I get extra towels?') and run again.")
