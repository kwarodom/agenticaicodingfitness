#!/usr/bin/env python3
"""Exercise 02 · Pick the right primitive.

Part A (offline, free): a six-question quiz — for each need, which primitive
fits: "choice", "noul" or "score"? Fill QUIZ below.

Part B (live): write three questions for hotel guest messages —
  department (choice) · mentions_refund (noul) · anger (score)

Run: .venv/bin/python week24/02_three_primitives/exercises/ex02_pick_the_primitive.py
Stuck? Compare with exercises/solutions/ex02_pick_the_primitive.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from jevkit import ask, banner, check, choice, noul, score, show  # noqa: E402,F401

# ── PART A ── TODO: replace each None with "choice", "noul" or "score" ────────
QUIZ = {
    "q1_which_language": None,   # Which ONE language is the message written in: en, th, zh, other?
    "q2_mentions_pool": None,    # Does the review mention the swimming pool at all?
    "q3_how_satisfied": None,    # How satisfied is the guest: very unhappy … delighted (5 levels)?
    "q4_tags": None,             # A review can praise food, staff AND location — tag each one (per tag)
    "q5_which_building": None,   # Which of our 3 hotel buildings (A, B, C, or unknown) is it about?
    "q6_how_severe": None,       # How severe is the reported leak: cosmetic … flooding?
}

# ── PART B ── TODO 1: a CHOICE with at least 3 departments + an escape option
department = None

# ── TODO 2: a NOUL — does the message ask for (or threaten to demand) a refund?
mentions_refund = None

# ── TODO 3: a SCORE — how angry is the guest? 3+ ordered, concrete levels, calmest first
anger = None

TEST_MESSAGES = [
    "I've been waiting 45 minutes for room service. If it isn't here in 10 I want my money back.",
    "Hi! Could housekeeping bring an extra blanket when convenient?",
    "The key card stopped working again. Third time today. Honestly ridiculous.",
]


# ─────────────────────────── checker — no need to edit below ────────────────
ANSWER_KEY = {"q1_which_language": "choice", "q2_mentions_pool": "noul",
              "q3_how_satisfied": "score", "q4_tags": "noul",
              "q5_which_building": "choice", "q6_how_severe": "score"}
WHY = {"q1_which_language": "exactly one of a fixed set → choice",
       "q2_mentions_pool": "a yes/no proposition → noul",
       "q3_how_satisfied": "an ordered scale → score",
       "q4_tags": "several labels can be true at once → one noul PER tag, not one choice",
       "q5_which_building": "one of a set, with an escape option → choice",
       "q6_how_severe": "ordered levels of severity → score"}


def main() -> None:
    banner("Exercise 02 · pick the primitive")
    print("▣ PART A · quiz (offline, $0)")
    if any(v is None for v in QUIZ.values()):
        print("✕ fill every QUIZ answer with 'choice', 'noul' or 'score' first")
        sys.exit(1)
    right = 0
    for k, v in QUIZ.items():
        good = str(v).strip().lower() == ANSWER_KEY[k]
        right += good
        check(good, f"{k}: {v}", f"{k}: you said {v!r} — {WHY[k]}")
    print(f"◆ quiz: {right}/{len(QUIZ)}")

    print("\n▣ PART B · question shapes (offline, $0)")
    ok = True
    crit = (department or {}).get("criteria") or {}
    ok &= check(isinstance(department, dict) and department.get("type") == "choice" and len(crit) >= 4
                and any(k in crit for k in ("unknown", "other", "not_stated")),
                f"department is a choice with {len(crit)} options incl. an escape option",
                "TODO 1: department = choice('…', {…3+ teams…, 'unknown': '…'})")
    ok &= check(isinstance(mentions_refund, dict) and mentions_refund.get("type") == "noul",
                "mentions_refund is a noul", "TODO 2: mentions_refund = noul('…')")
    levels = (anger or {}).get("criteria") or []
    ok &= check(isinstance(anger, dict) and anger.get("type") == "score" and len(levels) >= 3,
                f"anger is a score with {len(levels)} levels", "TODO 3: anger = score('…', [calm, …, furious])")
    if right < len(QUIZ) or not ok:
        print("\n⚠ fix the ✕ lines above and run again — no API call was made.")
        sys.exit(1)

    questions = {"department": department, "mentions_refund": mentions_refund, "anger": anger}
    print(f"\n▣ PART C · asking Jev about {len(TEST_MESSAGES)} messages (one batched call each)")
    for msg in TEST_MESSAGES:
        print(f"\n━━ “{msg}”")
        show(ask({"message": msg}, questions), top=3)
    print("\n═ Did the key-card message get a high anger score but a low refund noul?")
    print("  That is two independent judgments working as designed.")


if __name__ == "__main__":
    main()
