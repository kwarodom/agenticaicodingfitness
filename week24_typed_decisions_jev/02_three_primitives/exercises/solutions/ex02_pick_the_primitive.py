#!/usr/bin/env python3
"""Solution · Exercise 02 — one good answer (yours may differ and still be right)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevkit import choice, noul, score  # noqa: E402

import ex02_pick_the_primitive as ex  # noqa: E402

ex.QUIZ.update({
    "q1_which_language": "choice",
    "q2_mentions_pool": "noul",
    "q3_how_satisfied": "score",
    "q4_tags": "noul",            # one noul per tag — they are not mutually exclusive
    "q5_which_building": "choice",
    "q6_how_severe": "score",
})

ex.department = choice("Which hotel team should handle `message`?", {
    "room_service": "Food or drink orders delivered to the room, and their delays",
    "housekeeping": "Cleaning, towels, blankets, linen and amenities",
    "front_desk": "Key cards, check-in/out, billing and general requests",
    "unknown": "No clear request for any of these teams",
})
ex.mentions_refund = noul(
    "Does `message` ask for a refund or say the guest wants their money back?",
    true="A refund, compensation or money back is requested or demanded, even conditionally",
    false="No refund or money back is mentioned",
)
ex.anger = score("How angry is the guest who wrote `message`?", [
    "Calm or friendly",
    "Mildly annoyed",
    "Clearly frustrated, complaining",
    "Furious: threats, insults or demands",
])

if __name__ == "__main__":
    ex.main()
