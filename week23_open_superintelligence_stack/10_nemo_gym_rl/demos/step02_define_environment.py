#!/usr/bin/env python3
"""PART 2 · Define an environment — a task + a deterministic verifier  [INTERMEDIATE]

A NeMo Gym environment is just a task generator plus a reward_fn. This chapter
shows a Python-ish environment for a tool-use / calculator task whose verifier
returns 1.0 on pass else 0.0 — and runs that verifier for real on canned rollouts.

Run:  python demos/step02_define_environment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

ENV_CODE = '''\
# 🖥️ a NeMo-Gym-style verifiable-reward environment (deterministic verifier)
class CalculatorEnv:
    """Task: 'what is A op B?'  Verifier: exact-match on the numeric answer."""

    def sample(self):                       # hand the policy a prompt
        return {"prompt": "What is 37 * 24? Answer with the number only.",
                "answer": "888"}

    def reward_fn(self, task, answer):      # score the policy's rollout
        got = answer.strip().split()[-1] if answer.strip() else ""
        return 1.0 if got == str(task["answer"]) else 0.0   # 1.0 pass / 0.0 fail
'''


def main() -> None:
    view.banner("PART 2", "Define an environment — task + verifier", "INTERMEDIATE")
    view.mode_line()

    print("A slice of NeMo Gym's verifiable-reward environments:\n")
    print(f"  {'Environment':<14}{'Verifier':<32}{'Reward':<20}task type")
    print("  " + "─" * 84)
    for name, verifier, reward, task in sim.ENVIRONMENTS:
        print(f"  {name:<14}{verifier:<32}{reward:<20}{task}")
    print()

    print("An environment is small — a task sampler + a reward_fn:\n")
    print(ENV_CODE)

    print("Deterministic verifier in action (this is sim.reward_fn scoring rollouts):\n")
    task = {"prompt": "book the 2pm meeting room", "answer": "reserve(14:00)"}
    for cand in ["reserve(14:00)", "reserve(15:00)", "I can help with that!"]:
        r = sim.reward_fn(task, cand)
        mark = "PASS ✓" if r == 1.0 else "fail ✗"
        print(f"  reward_fn(gold={task['answer']!r:<18} answer={cand!r:<22}) → {r:.1f}  {mark}")
    print()

    print("Why DETERMINISTIC matters: the same answer always gets the same reward, so")
    print("the training signal is stable and can't be gamed by a fuzzy judge. The verifier")
    print("can be exact-match (math), a unit-test runner (code), or a JSON tool-call check.")

    print("\nTakeaway: an environment = a task + a reward_fn that returns a scalar. Next:")
    print("feed a GROUP of rollouts through it and run the GRPO training loop.")


if __name__ == "__main__":
    main()
