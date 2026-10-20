#!/usr/bin/env python3
"""Lab 02-3 · Score — a position on YOUR ordered scale.

A Score returns an expected value across levels 0…n-1 (a probability-weighted
average), plus the probability of each level and a `legend` echoing your text.

  • levels start at index 0
  • 1.6 means "mostly between level 1 and 2" — it is NOT a measurement
  • write each level as a concrete situation, lowest first
  • use it for thresholds ("≥ 2 means escalate"), never to interpolate numbers

Run: .venv/bin/python week24/02_three_primitives/labs/lab03_score.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from jevkit import ask, banner, score, show, step, table, validate  # noqa: E402

# Concrete levels: each describes a SITUATION a person could recognise,
# not a vague word like "medium".
URGENCY = {"urgency": score("How urgent is the request in `message`?", [
    "No time pressure: a question or a request for later",
    "Should be handled today, guest is mildly inconvenienced",
    "Needs action within the hour: guest cannot use part of the room",
    "Emergency now: safety risk, flooding, fire, or someone hurt",
])}

MESSAGES = [
    "Could you recommend a restaurant for tomorrow night?",
    "The bedside lamp bulb is out, whenever you get a chance.",
    "There's no hot water and I have a meeting in 40 minutes.",
    "Water is pouring from the ceiling onto the electrical sockets!",
]

banner("Lab 02-3 · score", "an expected position on your ordered levels")

step(1, "the full answer for one message")
resp = ask({"message": MESSAGES[2]}, URGENCY)
a = validate(resp, URGENCY)["urgency"]
show(resp)
# The expected value is just Σ level × probability. Recompute it yourself:
expected = sum(int(k) * p for k, p in a["probabilities"].items())
print(f"◆ Σ level×p = {expected:.2f}  vs API score {a['score']:.2f}  (same idea, rounded)")

step(2, "four messages across the scale")
rows = []
for msg in MESSAGES:
    r = ask({"message": msg}, URGENCY, quiet=True)
    s = validate(r, URGENCY)["urgency"]
    top_level = max(s["probabilities"], key=s["probabilities"].get)
    action = "page duty manager" if s["score"] >= 2.5 else "same-hour ticket" if s["score"] >= 1.5 else "normal queue"
    rows.append([msg, f"{s['score']:.2f}", top_level, f"{s['confidence']:.2f}", action])
table(rows, ["message", "score", "top level", "conf", "→ policy (in code)"])

print("\n═ Thresholds like 'score ≥ 2.5 → page' live in YOUR code, so you can tune")
print("  them later without re-asking the model. Don't read 1.6 as '1.6 hours'.")
print("═ execute: False — no one was paged; this lab prints proposals only.")
