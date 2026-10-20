#!/usr/bin/env python3
"""PART 4 · Evaluate — score task success  [ADVANCED]

You can only optimize goodput if you can MEASURE success. This runs a small GOLDEN
SET through an LLM-as-JUDGE:

    task → agent answer → judge verdict (pass/fail) → aggregate to a score

— the exact pattern from Weeks 10 & 15 that gates quality in CI. Without evaluation,
chasing cheaper tokens silently buys worse outcomes. In REAL the judge is your live
endpoint scoring one case; the rest of the golden set uses recorded verdicts so the
demo stays fast. In SIM every verdict is a recorded constant.

Run:  python demos/step04_evaluate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402

# a tiny golden set: (task, agent answer, expected, recorded verdict)
GOLDEN = [
    ("What color is the sky on a clear day?", "Blue.", "blue", True),
    ("2 + 2 = ?", "4", "4", True),
    ("Capital of France?", "Paris", "paris", True),
    ("Is 17 prime?", "No, 17 = 3 × 6.", "yes", False),
    ("JSON key for a user's age?", '{"age": 30}', "age", True),
]


def _bar(v: float, width: int = 24) -> str:
    fill = int(round(v * width))
    return "█" * fill + "·" * (width - fill)


def main() -> None:
    view.banner("PART 4", "Evaluate — score task success", "ADVANCED")
    view.mode_line()

    print("Performance is meaningless without CORRECTNESS. An LLM-as-judge scores each")
    print("answer against a golden set — the CI gate from Weeks 10 & 15:\n")
    print("    task → agent answer → judge: pass/fail → aggregate to a score\n")

    # In REAL, let the live endpoint judge the first case for real.
    live_verdict = None
    if not view.is_sim():
        task, answer, expected, _ = GOLDEN[0]
        got = view.classify(
            f"You are a strict grader. Task: {task}\nAnswer given: {answer}\n"
            f"Grade the answer: PASS or FAIL.",
            labels=["PASS", "FAIL"], title="LLM-as-judge scoring one golden case")
        live_verdict = got == "pass"
        print()

    print(f"  {'task':<40}{'verdict':>9}")
    print("  " + "─" * 52)
    passes = 0
    for i, (task, _answer, _expected, verdict) in enumerate(GOLDEN):
        if i == 0 and live_verdict is not None:
            verdict = live_verdict
        passes += int(verdict)
        mark = "PASS ✓" if verdict else "FAIL ✗"
        print(f"  {task[:38]:<40}{mark:>9}")
    score = passes / len(GOLDEN)
    print("  " + "─" * 52)
    print(f"  {'GOLDEN-SET SCORE':<40}{score:>8.0%}")
    print(f"  {_bar(score)}\n")

    GATE = 0.80
    if score >= GATE:
        print(f"  [PASS] score {score:.0%} ≥ gate {GATE:.0%} — safe to merge / promote.\n")
    else:
        print(f"  [FAIL] score {score:.0%} < gate {GATE:.0%} — block the merge (regression).\n")

    print("This is the missing half of performance: cheaper or faster tokens are only a WIN")
    print("if this score holds. Wire it into CI so a prompt/model change can't silently")
    print("regress quality — cost-per-success only means something once success is measured.\n")

    print("Takeaway: evaluation gates goodput. perf (Dynamo) × correctness (this) = value.")
    print("That's the economics of intelligence — see the appendix.")


if __name__ == "__main__":
    main()
