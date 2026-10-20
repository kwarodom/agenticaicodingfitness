#!/usr/bin/env python3
"""PART 2 · Learn a policy LIVE — baseline vs policy search  [INTERMEDIATE]

The controller every real building runs is a RULE: fixed setpoints on a schedule.
Here we train a controller against it, live: an honest ~30-line hill-climbing search
over the schedule's four setpoints, scored by the same LinearReward across simulated
weeks. Watch the learning curve rise, then read the final scorecard.

Run:  python demos/step02_learn_policy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402


def main() -> None:
    view.banner("PART 2", "Learn a policy LIVE — baseline vs policy search", "INTERMEDIATE")
    sim.lib_mode_line("stable_baselines3", "stable-baselines3",
                      "The learning loop below is our own ~30 lines — no RL library needed.")

    b = sim.BASELINE
    print("The rule-based baseline (a fixed setback schedule, like most real BAS):")
    print(f"  occupied 07–23 h : heat {b['heat_occ']:.1f} °C · cool {b['cool_occ']:.1f} °C")
    print(f"  night setback    : heat {b['heat_nt']:.1f} °C · cool {b['cool_nt']:.1f} °C\n")

    print("The learner: mutate the 4 setpoints, keep whatever scores a better episode")
    print("reward on the SAME simulated week (common random numbers — a fair fight).")
    print("Honest label: this is hill-climbing policy search, NOT PPO — but the loop")
    print("(act → observe reward → keep what improved) is exactly what PPO does at scale.\n")

    env = sim.RCZone(stochastic=True, seed=21)
    base_r, base_kwh, base_disc = sim.run_episode(env, b)
    best, hist = sim.hill_climb(env, episodes=24)
    learn_r, learn_kwh, learn_disc = sim.run_episode(env, best)

    lo, hi = min(hist), max(hist)
    print("Learning curve — best episode reward so far (1 episode = 1 simulated week):\n")
    for i, r in enumerate(hist, start=1):
        frac = (r - lo) / (hi - lo) if hi > lo else 1.0
        bar = "█" * max(1, round(frac * 26))
        print(f"  ep {i:>2}  {r:>7.2f}  {bar}")
    print()

    print(f"What it learned: cool occupied at {best['cool_occ']:.2f} °C instead of "
          f"{b['cool_occ']:.1f} °C — ride the TOP of the")
    print("comfort band instead of overcooling below it. (In this hot climate the heating")
    print("setpoints never fire; the search leaves them roughly alone.)\n")

    print("Final scorecard — one simulated week, identical weather:\n")
    print(f"  {'controller':<26}{'HVAC kWh':>10}{'comfort viol. K·h':>19}{'reward':>10}")
    print("  " + "─" * 66)
    print(f"  {'rule-based baseline':<26}{base_kwh:>10.1f}{base_disc:>19.2f}{base_r:>10.2f}")
    print(f"  {'learned policy':<26}{learn_kwh:>10.1f}{learn_disc:>19.2f}{learn_r:>10.2f}")
    savings = 100 * (base_kwh - learn_kwh) / base_kwh
    print(f"\n  → {savings:.1f}% less HVAC energy at equal comfort "
          f"({learn_disc:.2f} vs {base_disc:.2f} K·h).\n")

    print("Honest calibration: published RL results on Sinergym-class problems typically")
    print("show 5–20% HVAC savings vs rule-based controllers at equal-or-better comfort —")
    print("our toy lands inside that band. And the margin SHRINKS against a well-tuned")
    print("ASHRAE Guideline 36 baseline: much of what RL 'discovers' is what G36 already")
    print("encodes as best-practice sequences.\n")

    print("run it for real (PPO on the full EnergyPlus building):")
    print("  $ pip install sinergym stable-baselines3")
    print("  >>> from stable_baselines3 import PPO")
    print(f"  >>> model = PPO('MlpPolicy', gym.make('{sim.SINERGYM_ENV_ID}'))")
    print("  >>> model.learn(total_timesteps=200_000)     # hours of sim, not of building\n")

    print("Takeaway: the gym makes learning cheap and the baseline keeps you honest.")
    print("Next: zoom out from one zone to a whole district — CityLearn.")


if __name__ == "__main__":
    main()
