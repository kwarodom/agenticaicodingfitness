#!/usr/bin/env python3
"""NeMo Data Flywheel simulator — learn the self-evolving loop with no GPU."""
from __future__ import annotations

import time

# The models the flywheel juggles: a big "teacher" and the small "student" it distills.
MODELS = [
    ("nemotron-3-super:120b-a12b", "teacher", "labels + judges"),
    ("nemotron-3-nano:30b-a3b",    "student", "the model you promote"),
]


_TOK = {"nemotron-3-super:120b-a12b": 20.0, "nemotron-3-nano:30b-a3b": 54.0}


def installed_models() -> list[str]:
    return [m for m, *_ in MODELS]


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40.0))


# One flywheel turn: observe → curate → customize → evaluate → promote.
# Each round the small student closes the gap to the teacher at a fraction of the cost.
def run_flywheel(rounds: int = 4):
    """Yield (round, student_acc, teacher_acc, cost_ratio, promoted)."""
    teacher = 0.91
    student = 0.62
    cost = 1.00  # student cost relative to teacher (starts cheap, stays cheap)
    for r in range(1, rounds + 1):
        student = round(student + (teacher - student) * 0.45, 3)
        cost = round(0.14 + 0.01 * r, 3)  # student runs at ~1/7th the teacher cost
        promoted = student >= teacher - 0.02
        yield r, student, teacher, cost, promoted


_CANNED = ("[simulated flywheel] Production logs become training data (Curator), a small "
           "student is fine-tuned to match a big teacher (Customizer), an LLM-judge proves "
           "it's as good at lower cost (Evaluator), and the winner is promoted — automatically, "
           "on your DGX. Observe → learn → optimize, on repeat.")


def stream_generate(prompt: str, model: str):
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(0.03)
