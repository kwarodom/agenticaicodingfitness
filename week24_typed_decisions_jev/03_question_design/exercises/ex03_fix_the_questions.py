#!/usr/bin/env python3
"""Exercise 03 · Fix the questions — and MEASURE the improvement.

Below are three deliberately bad questions, each with a classic design flaw:
  • bad_outage   — vague word ("bad") instead of the exact condition
  • bad_team     — no escape option, so "thanks!" must be forced into a team
  • bad_callback — criteria contradict the instruction (true means NO)

Write a good version of each (TODO 1-3). The checker asks the bad AND the good
questions in the same batched call per message (they are independent), then
scores both against the human labels. Aim: good ≥ bad on every question.

Run: .venv/bin/python week24/03_question_design/exercises/ex03_fix_the_questions.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from jevkit import ask, banner, check, choice, noul, step, table, validate  # noqa: E402,F401

TEAMS = ["hvac", "electrical", "plumbing", "not_a_request"]

# Human labels: (message, outage_now?, team, wants_callback?)
TEST_SET = [
    ("The chiller just tripped and the building is heating up. Call me on 081-555-0101.", True, "hvac", True),
    ("Lights on floor 3 have been flickering all week, can you check when you're next on site?", False, "electrical", False),
    ("A pipe burst in the basement plant room and it's flooding right now!", True, "plumbing", False),
    ("Thanks for the great service visit yesterday!", False, "not_a_request", False),
    ("Could someone ring me to discuss the AHU filter replacement schedule?", False, "hvac", True),
    ("Main breaker for the kitchen has tripped, no power to the ovens.", True, "electrical", False),
]

# ── the BAD questions (do not edit) ──────────────────────────────────────────
BAD = {
    "bad_outage": noul("Is this bad?"),
    "bad_team": choice("Which team?", {"hvac": None, "electrical": None, "plumbing": None}),
    "bad_callback": noul(
        "Does the sender of `message` want a phone call?",
        true="The sender does NOT ask to be called",     # ← contradicts the instruction
        false="The sender asks to be called",
    ),
}

# ── TODO 1 ── good_outage: a NOUL with the exact condition.
#   Hint: "current failure happening now" vs "a problem to look at later".
good_outage = None

# ── TODO 2 ── good_team: a CHOICE whose options are exactly TEAMS (above),
#   each with a description — including "not_a_request".
good_team = None

# ── TODO 3 ── good_callback: a NOUL whose criteria AGREE with the instruction
#   (true = yes, they want a call).
good_callback = None


# ─────────────────────────── checker — no need to edit below ────────────────
def main() -> None:
    banner("Exercise 03 · fix the questions")
    step(1, "shape checks (offline, $0)")
    ok = True
    ok &= check(isinstance(good_outage, dict) and good_outage.get("type") == "noul"
                and good_outage.get("instructions") != BAD["bad_outage"]["instructions"],
                "good_outage is a new noul", "TODO 1: good_outage = noul('…exact condition…')")
    ok &= check(isinstance(good_team, dict) and good_team.get("type") == "choice"
                and set((good_team.get("criteria") or {})) == set(TEAMS)
                and all((good_team.get("criteria") or {}).values()),
                "good_team has exactly the 4 TEAMS options, each described",
                f"TODO 2: good_team options must be {TEAMS}, each with a description")
    crit = (good_callback or {}).get("criteria") or {}
    ok &= check(isinstance(good_callback, dict) and good_callback.get("type") == "noul"
                and "NOT" not in str(crit.get("true", "")),
                "good_callback is a noul whose 'true' is not a negation",
                "TODO 3: good_callback = noul('…', true='wants a call', false='no call requested')")
    if not ok:
        print("\n⚠ fix the ✕ lines above and run again — no API call was made.")
        sys.exit(1)

    questions = dict(BAD, good_outage=good_outage, good_team=good_team, good_callback=good_callback)
    step(2, f"bad vs good on {len(TEST_SET)} labeled messages (one batched call each)")
    score = {k: 0 for k in questions}
    clarity = {k: 0.0 for k in questions}          # nouls: how far from 0.5 (1 = decisive, 0 = coin flip)
    rows = []
    for msg, outage, team, callback in TEST_SET:
        a = validate(ask({"message": msg}, questions, quiet=True), questions)
        score["bad_outage"] += (a["bad_outage"]["noul"] >= .5) == outage
        score["good_outage"] += (a["good_outage"]["noul"] >= .5) == outage
        score["bad_team"] += a["bad_team"]["choice"] == team
        score["good_team"] += a["good_team"]["choice"] == team
        score["bad_callback"] += (a["bad_callback"]["noul"] >= .5) == callback
        score["good_callback"] += (a["good_callback"]["noul"] >= .5) == callback
        for k in ("bad_outage", "good_outage", "bad_callback", "good_callback"):
            clarity[k] += abs(a[k]["noul"] - 0.5) * 2 / len(TEST_SET)
        rows.append([msg, a["bad_team"]["choice"], a["good_team"]["choice"], team])
    table(rows, ["message", "bad_team", "good_team", "label"])

    step(3, "accuracy before → after")
    n = len(TEST_SET)
    for name in ("outage", "team", "callback"):
        b, g = score[f"bad_{name}"], score[f"good_{name}"]
        extra = ""
        if f"bad_{name}" in clarity and name != "team":
            extra = f"   clarity {clarity[f'bad_{name}']:.2f} → {clarity[f'good_{name}']:.2f}"
        check(g >= b, f"{name:<9} {b}/{n} → {g}/{n}{extra}",
              f"{name:<9} {b}/{n} → {g}/{n}{extra}  (your version is worse — rethink it)")
    print("\n◆ clarity = average distance of a noul from 0.5 (1.00 = always decisive, 0 = coin flips)")
    print("═ Accuracy can tie while clarity differs: a contradictory or vague question leaves")
    print("  answers closer to 0.5, so fewer of them clear your confidence thresholds.")


if __name__ == "__main__":
    main()
