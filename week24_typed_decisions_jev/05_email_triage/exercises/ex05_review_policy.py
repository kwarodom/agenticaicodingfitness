#!/usr/bin/env python3
"""Exercise 05 · Write the inbox review policy yourself.

Jev gives you judgments. YOU decide what happens. Implement route(answers) so
that it returns one of: "support", "sales", "finance", "hr", "newsletter",
"human_review", or "fraud_desk".

Rules to implement (in this priority order):
  1. suspicious >= 0.5            → "fraud_desk"     (a person checks it; nothing is paid)
  2. category is "other"          → "human_review"
  3. category confidence < 0.75   → "human_review"
  4. urgent >= 0.5                → "human_review"   (a person must see outages fast)
  5. otherwise                    → the category label itself

Run:  .venv/bin/python week24/05_email_triage/exercises/ex05_review_policy.py
The offline tests run first (free). Only when they all pass does the live inbox run.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE.parents[1] / "labs"))
from jevkit import ask, banner, check, step, table, validate  # noqa: E402


def route(answers: dict) -> str:
    """answers = {"category": {"choice", "confidence", ...}, "urgent": {"noul"}, ...}"""
    # ── TODO ── replace this line with rules 1-5 from the docstring above.
    return "TODO"


# ─────────────────────────── checker — no need to edit below ────────────────
def fake(category="support", conf=0.95, urgent=0.1, sensitive=0.1, suspicious=0.05, reply=0.8) -> dict:
    """A hand-made answer dict shaped exactly like Jev's — for free offline tests."""
    return {"category": {"type": "choice", "choice": category, "confidence": conf,
                         "probabilities": {category: 1.0}},
            "urgent": {"type": "noul", "noul": urgent}, "sensitive": {"type": "noul", "noul": sensitive},
            "suspicious": {"type": "noul", "noul": suspicious}, "reply_needed": {"type": "noul", "noul": reply}}


TESTS = [
    ("clear sales email", fake("sales"), "sales"),
    ("fraud beats a confident finance label", fake("finance", conf=0.99, suspicious=0.9), "fraud_desk"),
    ("'other' always goes to a person", fake("other", conf=0.99), "human_review"),
    ("low confidence goes to a person", fake("support", conf=0.6), "human_review"),
    ("urgent outage goes to a person", fake("support", urgent=0.93), "human_review"),
    ("suspicious 0.49 is below the fraud line", fake("newsletter", suspicious=0.49), "newsletter"),
    ("fraud check runs before 'other'", fake("other", suspicious=0.7), "fraud_desk"),
]


def main() -> None:
    banner("Exercise 05 · my review policy")
    step(1, "offline tests — free, deterministic, no API call")
    passed = 0
    for name, answers, want in TESTS:
        got = route(answers)
        passed += check(got == want, f"{name} → {got}", f"{name}: expected {want!r}, got {got!r}")
    if passed < len(TESTS):
        print(f"\n⚠ {passed}/{len(TESTS)} tests pass — fix route(), save, run again. No API call was made.")
        sys.exit(1)

    step(2, "all tests pass — now run YOUR policy on the live inbox")
    from lab01_triage_inbox import QUESTIONS, load_inbox
    rows = []
    for e in load_inbox():
        a = validate(ask(e["state"], QUESTIONS, quiet=True), QUESTIONS)
        rows.append([e["id"], e["state"]["subject"][:30], a["category"]["choice"],
                     f"{a['suspicious']['noul']:.2f}", f"{a['urgent']['noul']:.2f}", route(a)])
    table(rows, ["id", "subject", "category", "suspic", "urgent", "YOUR route"])
    print("═ execute: False · no_send: True · no_delete: True — your policy proposed, nobody acted.")


if __name__ == "__main__":
    main()
