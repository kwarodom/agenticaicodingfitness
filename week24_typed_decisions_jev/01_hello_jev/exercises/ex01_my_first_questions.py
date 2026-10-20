#!/usr/bin/env python3
"""Exercise 01 · Write your first three questions.

Fill in the three TODOs, save, then run:
    .venv/bin/python week24/01_hello_jev/exercises/ex01_my_first_questions.py

The checker validates your question SHAPES first (free, no API call), then asks
Jev about three test messages. Stuck? Compare with exercises/solutions/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from jevkit import ask, banner, check, choice, noul, score, show  # noqa: E402,F401

TEST_MESSAGES = [
    "The shower in my bathroom has no hot water. This is unacceptable.",
    "Could someone please bring two extra pillows to the bedroom? Thank you so much!",
    "What time does breakfast start?",
]

# ── TODO 1 ── a NOUL: is the message a complaint?
#   noul("…question about `message`…")
is_complaint = None

# ── TODO 2 ── a CHOICE: which area of the room is the message about?
#   Use at least: bathroom, bedroom, balcony — PLUS an escape option for
#   messages that mention no area (name it e.g. "not_stated").
#   choice("…question…", {"bathroom": "…", …})
room_area = None

# ── TODO 3 ── a SCORE: how polite is the message? At least 3 ORDERED levels,
#   lowest first, e.g. ["Rude or hostile", "Neutral", "Very polite"].
politeness = None


# ─────────────────────────── checker — no need to edit below ────────────────
def main() -> None:
    banner("Exercise 01 · my first questions")
    ok = True
    ok &= check(isinstance(is_complaint, dict) and is_complaint.get("type") == "noul"
                and bool(is_complaint.get("instructions")),
                "is_complaint is a noul with instructions",
                "TODO 1: is_complaint should be noul('…')")
    crit = (room_area or {}).get("criteria") or {}
    escape = [k for k in crit if k in ("not_stated", "unknown", "none", "other", "no_area")]
    ok &= check(isinstance(room_area, dict) and room_area.get("type") == "choice"
                and {"bathroom", "bedroom", "balcony"} <= set(crit) and bool(escape),
                f"room_area is a choice with {len(crit)} options (includes an escape option)",
                "TODO 2: room_area needs bathroom, bedroom, balcony AND an escape option like not_stated")
    levels = (politeness or {}).get("criteria") or []
    ok &= check(isinstance(politeness, dict) and politeness.get("type") == "score"
                and isinstance(levels, list) and len(levels) >= 3,
                f"politeness is a score with {len(levels)} ordered levels",
                "TODO 3: politeness should be score('…', [lowest, …, highest]) with 3+ levels")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again — no API call was made.")
        sys.exit(1)

    questions = {"is_complaint": is_complaint, "room_area": room_area, "politeness": politeness}
    print(f"\n▣ asking Jev about {len(TEST_MESSAGES)} test messages …")
    for msg in TEST_MESSAGES:
        print(f"\n━━ “{msg}”")
        show(ask({"message": msg}, questions), top=3)
    print("\n═ Look at the breakfast message: which room_area won? If it was not your escape")
    print("  option, sharpen that option's description and run again.")


if __name__ == "__main__":
    main()
