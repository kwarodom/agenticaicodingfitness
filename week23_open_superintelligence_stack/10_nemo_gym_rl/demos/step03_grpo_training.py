#!/usr/bin/env python3
"""PART 3 · GRPO training loop on the DGX  [ADVANCED]

GRPO (Group Relative Policy Optimization): for each prompt, sample a GROUP of
rollouts, score each with the verifier, use the group MEAN as the baseline to
compute advantages, and update the policy toward high-advantage rollouts. No
separate value network. This chapter runs a simulated training run over N rounds
with mean reward climbing.

Run:  python demos/step03_grpo_training.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import sim  # noqa: E402
import view  # noqa: E402

LAUNCH = """\
# 🖥️ on the DGX — post-train a Nemotron policy with NeMo RL (GRPO)
uv run nemo_rl.grpo \\
  policy.model=nvidia/nemotron-3-nano \\
  env=nemo_gym:math_verify,code_unit,tool_call \\
  grpo.num_rollouts_per_prompt=8 \\
  grpo.rounds=20 \\
  cluster.gpus=1                 # 1 DGX Spark; use cluster.tp=2 across 2 Sparks
"""


def _bar(v: float, width: int = 32) -> str:
    fill = int(round(v * width))
    return "█" * fill + "·" * (width - fill)


def main() -> None:
    view.banner("PART 3", "GRPO training loop on the DGX", "ADVANCED")
    view.mode_line()

    print("How GRPO uses a GROUP (no value network, no separate critic):")
    print("  1. ROLLOUT   sample G=8 completions for one prompt")
    print("  2. REWARD    verifier scores each → e.g. [0,0,1,0,1,0,0,1]")
    print("  3. BASELINE  advantage = reward − group mean  (the 'group-relative' part)")
    print("  4. UPDATE    policy-gradient step toward positive-advantage rollouts\n")

    grp = sim.rollout_group(round_idx=6, group_size=8)
    adv = sim.advantages(grp)
    mean = sum(grp) / len(grp)
    print("One prompt's group at round 6 (this is sim.rollout_group / sim.advantages):")
    print(f"  rewards    = {grp}")
    print(f"  group mean = {mean:.3f}   (the GRPO baseline)")
    print(f"  advantages = {adv}\n")

    print("Launch it for real on the DGX:\n")
    print(LAUNCH)

    print("Simulated training run — mean verifier reward per round:\n")
    curve = sim.reward_curve(rounds=20, start=0.30, end=0.80)
    for i, r in enumerate(curve):
        if i % 2 == 0 or i == len(curve) - 1:      # print every other round
            print(f"  round {i:>2}  reward {r:0.2f}  {_bar(r)}")
    print(f"\n  base {curve[0]:.2f} → trained {curve[-1]:.2f}  (mean verifier reward)\n")

    print(f"Scaling on DGX Spark ({config.MODEL}):")
    print("  • 1 Spark  → small policy (Nemotron Nano), fewer parallel rollouts.")
    print("  • 2 Sparks → link over QSFP 200GbE, tensor-parallel TP=2 for a bigger")
    print("    policy or more rollouts/round — same NeMo RL config, just cluster.tp=2.")

    print("\nTakeaway: GRPO scores a GROUP, subtracts the mean, and climbs the reward")
    print("curve — no reward model, no critic. Next: many environments at once + eval.")


if __name__ == "__main__":
    main()
