#!/usr/bin/env python3
"""Solution · Exercise 01 — one good answer (yours may differ and still be right)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevkit import choice, noul, score  # noqa: E402

import ex01_my_first_questions as ex  # noqa: E402

ex.is_complaint = noul(
    "Does `message` complain about a problem or express dissatisfaction?",
    true="The guest reports something wrong or is unhappy",
    false="A neutral question or a polite request with no problem reported",
)
ex.room_area = choice("Which area of the guest room is `message` about?", {
    "bathroom": "Shower, toilet, sink, bath, hot water, towels in the bathroom",
    "bedroom": "Bed, pillows, sheets, wardrobe, desk, room temperature in the bedroom",
    "balcony": "Balcony door, balcony furniture, the view outside",
    "not_stated": "The message does not mention any specific area of the room",
})
ex.politeness = score("How polite is the tone of `message`?", [
    "Rude or hostile",
    "Blunt or demanding, but not rude",
    "Neutral",
    "Polite and friendly",
])

if __name__ == "__main__":
    ex.main()
