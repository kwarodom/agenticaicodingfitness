#!/usr/bin/env python3
"""PART 3 · Customize — distill a big teacher into a small student  [ADVANCED]

NeMo Customizer fine-tunes a SMALL model (the student) on the curated data so it
matches a BIG model (the teacher) on your domain — at a fraction of the serving cost.
LoRA / SFT / DPO / GRPO are the tools; distillation is the goal.

Run:  python demos/step03_customize.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402

METHODS = [
    ("SFT",  "supervised fine-tune on curated (prompt→good answer) pairs",  "baseline distill"),
    ("LoRA", "low-rank adapters — cheap, fast, swappable; fits 1 DGX Spark", "most common"),
    ("DPO",  "prefer chosen over rejected (from LLM-judge labels)",          "align to taste"),
    ("GRPO", "RL with verifiable rewards (see App 10 · NeMo Gym)",            "outcome-driven"),
]

CODE = """\
# NeMo Customizer (sketch) — distill Super(120B) → Nano(30B) on your DGX
from nemo_customizer import Customizer
job = Customizer(
    student="nemotron-3-nano:30b-a3b",
    teacher="nemotron-3-super:120b-a12b",   # provides labels / soft targets
    method="lora", r=16, alpha=32,
    dataset="curated/agent-62k",            # from Curator (Part 2)
).launch()                                  # 1 Spark; 2 Sparks (TP=2) for bigger students
"""


def main() -> None:
    view.banner("PART 3", "Customize — distill teacher into student", "ADVANCED")
    view.mode_line()

    print("Pick a customization method (all supported by NeMo Customizer):\n")
    print(f"  {'method':<7}{'what it does':<55}when")
    print("  " + "─" * 84)
    for m, what, when in METHODS:
        print(f"  {m:<7}{what:<55}{when}")
    print()
    print("Distillation — teach a small model to imitate a big one on YOUR domain:\n")
    print(CODE)
    print("On the hardware:")
    print("  • 1 DGX Spark (GB10, 128 GB): LoRA/SFT on Nano-class students.")
    print("  • 2 DGX Sparks over QSFP 200GbE (TP=2): larger students / faster epochs.\n")

    print("The point of distillation: the student ends up ~as good as the teacher on your")
    print("tasks, but serves at a fraction of the cost — which is what makes it promotable.\n")

    print("Takeaway: Customizer compresses a big model's domain skill into a small, cheap")
    print("one. Next: prove the student actually won, then promote it.")


if __name__ == "__main__":
    main()
