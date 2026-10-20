#!/usr/bin/env python3
"""PART 4 · Evaluate + promote — close the loop  [ADVANCED]

A model is only promoted if it earns it. NeMo Evaluator uses an LLM-judge to A/B the
student against the teacher; if the student matches quality at lower cost, it's
promoted to serve — and the flywheel spins again. This demo runs the loop.

Run:  python demos/step04_evaluate_promote.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402


def main() -> None:
    view.banner("PART 4", "Evaluate + promote — close the loop", "ADVANCED")
    view.mode_line()

    print("NeMo Evaluator (LLM-judge) A/B-tests the student vs the teacher each round.")
    print("Promote the student when it reaches teacher-quality at a fraction of the cost:\n")

    print(f"  {'round':<7}{'student acc':>12}{'teacher acc':>13}{'student $/tok':>15}{'':>4}verdict")
    print("  " + "─" * 74)
    for r, student, teacher, cost, promoted in sim.run_flywheel(rounds=4):
        gap = "█" * max(0, round(student * 24))
        verdict = "✅ PROMOTE" if promoted else "…keep teacher"
        print(f"  {r:<7}{student:>11.0%}{teacher:>13.0%}{cost:>14.2f}x   {verdict}")
        print(f"  {'':<7}{gap}")
    print()
    print("What just happened: over 4 turns the small student closed the quality gap to a")
    print("120B teacher while serving at ~1/7th the cost — so it gets promoted to production.\n")

    print("The self-evolving payoff (why sovereignty + the flywheel compound):")
    print("  • cheaper: serve a 30B student instead of a 120B teacher.")
    print("  • better: it learned from YOUR real traffic, not generic web data.")
    print("  • private: every step ran on your DGX — logs never left the perimeter.\n")

    view.generate("In two sentences, why does a data flywheel make a sovereign agent get "
                  "cheaper and better over time?", max_tokens=300, title="the promoted student")

    print("\nTakeaway: evaluate-then-promote is the ratchet that makes evolution safe —")
    print("only proven winners ship. Next app (4 · Dynamo): serve the promoted model at scale.")


if __name__ == "__main__":
    main()
