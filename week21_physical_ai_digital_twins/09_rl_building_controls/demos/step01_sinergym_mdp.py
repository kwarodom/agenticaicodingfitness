#!/usr/bin/env python3
"""PART 1 · Building control as an MDP — Sinergym  [BEGINNER]

An RL agent can't learn on the real building (a bad week of exploration = a bad week
for the guests). Sinergym (Univ. of Granada SAIL) wraps EnergyPlus in a gym: the sim
becomes an ENVIRONMENT the agent pokes with actions and scores with a reward. This
demo builds the exact same MDP on our tiny RC hotel zone and steps it live.

Run:  python demos/step01_sinergym_mdp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

DIAGRAM = """\
   AGENT ──── action: [heating sp, cooling sp] ────► GYM WRAPPER ──► EnergyPlus
     ▲                                                (Sinergym)      (App 6's sim)
     └──── obs (≈17 floats) + reward (LinearReward) ◄─────┘   threads + queues
"""


def main() -> None:
    view.banner("PART 1", "Building control as an MDP — Sinergym", "BEGINNER")
    sim.lib_mode_line("sinergym", "sinergym", "This demo's MDP runs on the built-in RC zone either way.")

    print("Sinergym v3+ drives EnergyPlus through its official Python API — App 6's")
    print("inversion-of-control lesson (E+ calls YOUR code back) turned into a gym with")
    print("threads and queues. The agent sees a plain reset()/step() loop:\n")
    print(DIAGRAM)

    print(f"Env IDs read building-climate-actionspace-stochastic:  {sim.SINERGYM_ENV_ID}")
    print(f"  • buildings: {', '.join(sim.SINERGYM_BUILDINGS)}")
    print(f"  • climates:  {', '.join(sim.SINERGYM_CLIMATES)}   · `-stochastic-` = weather noise")
    print("  • plus wrappers (normalize / log / WandB), stable-baselines3 integration,")
    print("    and a Docker image with EnergyPlus preinstalled.\n")

    print("The MDP, exactly (5Zone defaults):")
    print("  • OBSERVATION  Box of configured E+ variables + time features — ≈17 floats")
    print("  • ACTION       Box(2,) = [heating sp 12–23.25 °C, cooling sp 21.75–30 °C]")
    print("                 (or a discrete space of 10 setpoint pairs)")
    print("  • REWARD       LinearReward:  r = −w·λE·power − (1−w)·λT·Σ|T − comfort bound|")
    print("                 w=0.5 · comfort ~20–23.5 °C winter / 23–26 °C summer\n")

    env = sim.RCZone(stochastic=True, seed=21)
    obs = env.reset()
    print("Same MDP on OUR RC hotel zone (Bangkok wing, 1R1C, Δt=15 min) — obs at reset:")
    for i in range(0, len(sim.OBS_NAMES), 2):
        pair = "  ".join(f"{sim.OBS_NAMES[j]:<28}= {obs[j]:>8.2f}"
                         for j in range(i, min(i + 2, len(sim.OBS_NAMES))))
        print(f"  {pair}")
    print()

    while obs[2] < 7.0:                              # fast-forward to occupancy
        obs, _, _, _ = env.step(sim.schedule_policy(sim.BASELINE, obs[2]))
    print("Fast-forwarded to 07:00 (occupied). Four steps with action [21.0, 25.0]:\n")
    print(f"  {'hour':>6}{'T_out':>8}{'T_zone':>8}{'HVAC W':>9}{'reward':>9}   decomposition")
    print("  " + "─" * 76)
    for _ in range(4):
        hour = obs[2]
        obs, r, _, info = env.step((21.0, 25.0))
        e_term = env.W * env.LAMBDA_E * info["power_w"]
        c_term = (1 - env.W) * env.LAMBDA_T * info["discomfort_k"]
        print(f"  {hour:>6.2f}{info['tout']:>8.2f}{info['t_zone']:>8.2f}"
              f"{info['power_w']:>9.0f}{r:>9.3f}   −{e_term:.3f} energy − {c_term:.3f} comfort")
    print()
    print("Reading it: the zone drifted to 26.4 °C overnight (night setback), so at 07:00")
    print("the agent pays BOTH terms — energy (the unit is flat out, capped) AND comfort")
    print("(T is above the 26 °C band) — until the pull-down finishes. That morning-recovery")
    print("trade-off is exactly what the reward makes the agent learn.\n")

    print("run it for real (the identical loop on a full EnergyPlus building):")
    print("  $ pip install sinergym            # or use its Docker image (E+ preinstalled)")
    print("  >>> import gymnasium as gym, sinergym")
    print(f"  >>> env = gym.make('{sim.SINERGYM_ENV_ID}')")
    print("  >>> obs, info = env.reset(); obs, r, term, trunc, info = env.step([21.0, 25.0])\n")

    print("Takeaway: a gym wrapper turns the App 6 simulation into a place where an agent")
    print("can fail a thousand times for free. Next: actually learn a policy, live.")


if __name__ == "__main__":
    main()
