#!/usr/bin/env python3
"""PART 1 · What is verifiable-reward RL?  [BEGINNER]

Agent = Model + Harness. This app is the LEARNING layer: NeMo Gym gives you
verifiable-reward ENVIRONMENTS (a task + a programmatic verifier that returns a
reward), and NeMo RL runs the training (GRPO) that turns those rewards into a
better policy. This chapter explains the idea and the loop.

Run:  python demos/step01_verifiable_reward.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402

LOOP = """\
        ┌──────────────────────────────────────────────────────────┐
        │  the verifiable-reward RL loop (one GRPO step)             │
        ├──────────────────────────────────────────────────────────┤
        │  1. ROLLOUT   policy answers a prompt (a GROUP of tries)   │
        │  2. REWARD    verifier scores each try   → 1.0 pass / 0.0  │
        │  3. ADVANTAGE reward − group mean (group-relative baseline)│
        │  4. UPDATE    push policy toward high-advantage rollouts   │
        │  └────────────────────── repeat over N rounds ────────────┘│
        └──────────────────────────────────────────────────────────┘
"""


def main() -> None:
    view.banner("PART 1", "What is verifiable-reward RL?", "BEGINNER")
    view.mode_line()

    print("The core idea — an RL *environment* = a TASK + a VERIFIER:")
    print("  • The task hands the policy a prompt (a math problem, a coding task…).")
    print("  • The verifier is PROGRAM code that scores the answer → a scalar reward.")
    print("  • No human in the loop, no reward model to train — just a check that runs.\n")

    print("Why outcome-based rewards beat human-preference rewards for agents:")
    print(f"  {'':<22}{'verifiable reward':<24}human preference (RLHF)")
    print("  " + "─" * 70)
    for k, a, b in [
        ("Signal",       "did it PASS? (unit test)", "which reply do I like more"),
        ("Cost/scale",   "free, runs millions/hr",   "slow, expensive labels"),
        ("Gaming",       "hard — tests are objective","easy — reward hacking"),
        ("Best for",     "code / math / tool-use",   "tone / style / helpfulness"),
    ]:
        print(f"  {k:<22}{a:<24}{b}")
    print()

    print("Two tools, two jobs (both run on your DGX, cloud cost $0):")
    print("  • NeMo Gym → the ENVIRONMENTS: tasks + verifiers that emit rewards.")
    print("  • NeMo RL  → the TRAINER: runs GRPO (Group Relative Policy Optimization)")
    print("               to post-train a Nemotron policy from those rewards.\n")

    print("The loop you'll run in the next chapters:\n")
    print(LOOP)

    print("Takeaway: verifiable-reward RL = 'reward is a program, not a person'. Next:")
    print("we define an environment with a real reward_fn.")


if __name__ == "__main__":
    main()
