#!/usr/bin/env python3
"""Lab 02-1 · Choice — pick ONE option, and read the whole distribution.

Three guest messages, one Choice question:
  1. a clear request            → almost all probability on one option
  2. a message with TWO needs   → probability splits between two options
  3. a message with NO request  → the escape option ("unknown") wins

Lesson: the winner alone hides information. The distribution tells you
whether the model is sure, torn between two, or saying "none of these".

Run: .venv/bin/python week24/02_three_primitives/labs/lab01_choice.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from jevkit import ask, banner, choice, show, step, table, top2, validate  # noqa: E402

# One Choice question. Each option has a description: the model reads the
# descriptions, not just the option names, so write them like a rubric.
QUESTIONS = {
    "department": choice("Which hotel team should handle `message`?", {
        "hvac": "Heating, cooling, ventilation or air-conditioning problems",
        "housekeeping": "Cleaning, towels, linen, amenities, room tidiness",
        "front_desk": "Bookings, billing, check-in, check-out and general questions",
        "unknown": "The message contains no request for any of these teams",
    }),
}

MESSAGES = {
    "clear":     "The air conditioner in room 804 is blowing warm air.",
    "two_needs": "Our room is too hot and we also need fresh towels, please.",
    "no_request": "Thanks, we had a lovely stay!",
}

banner("Lab 02-1 · choice", "one option wins — but read the whole distribution")

rows = []
for i, (name, msg) in enumerate(MESSAGES.items(), 1):
    step(i, f"{name}: “{msg}”")
    resp = ask({"message": msg}, QUESTIONS)
    a = validate(resp, QUESTIONS)["department"]
    show(resp)
    p1, p2 = top2(a["probabilities"])
    rows.append([name, a["choice"], f"{p1:.2f}", f"{p2:.2f}", f"{p1 - p2:.2f}", f"{a['confidence']:.2f}"])

step(4, "side by side — the margin between the top two options is the tell")
table(rows, ["message", "winner", "top p", "2nd p", "margin", "confidence"])

print("\n═ A small margin means 'torn between two'. Your code can ask a human,")
print("  or split the message, instead of trusting the winner blindly.")
print("═ The escape option ('unknown') is what lets Jev say 'none of these'.")
