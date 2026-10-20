#!/usr/bin/env python3
"""Solution · Exercise 12 — plug the reference copilot's pieces into the exercise."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))           # copilot_reference.py

import copilot_reference as ref       # noqa: E402
import ex12_build_the_copilot as ex   # noqa: E402

ex.QUESTIONS = ref.QUESTIONS           # TODO 1 — five questions, guarded
ex.extract_room = ref.extract_room     # TODO 2 — keyword regex, then a lone 3-4 digit number
ex.policy = ref.policy                 # TODO 3 — safety override → gate → priority


def build_ticket(message, answers, decision, room, resp):   # TODO 4 — the exercise's 5-arg signature
    ticket = ref.build_ticket(message, answers, decision, room, ref.detect_language(message), resp)
    ticket["model"] = resp["model"]
    return ticket


ex.build_ticket = build_ticket

if __name__ == "__main__":
    ex.main()
