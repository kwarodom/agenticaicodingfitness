#!/usr/bin/env python3
"""PART 4 · BOPTEST — fair controller benchmarking over REST  [ADVANCED]

Sinergym lets you TRAIN; BOPTEST (IBPSA Project 1) lets you PROVE. High-fidelity
Modelica emulator buildings (Buildings library) with baseline controllers ship in
Docker and are driven via REST — deliberately framework-agnostic. Fixed scenarios +
fixed KPIs = results anyone can compare. This demo simulates the REST wire-trace and
scores our Ch 3 policy against the rule-based baseline on the standard KPIs.

Run:  python demos/step04_boptest_kpis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402


def main() -> None:
    view.banner("PART 4", "BOPTEST — fair controller benchmarking over REST", "ADVANCED")
    view.mode_line()

    print("Test cases (each a Modelica emulator + a baseline controller, in Docker):")
    print("  " + " · ".join(sim.BOPTEST_CASES) + "\n")

    print("The REST loop, wire-traced (values from our RC zone; BOPTEST speaks Kelvin):\n")
    print("  💻 controller                                        🖥️ BOPTEST (Docker)")
    print("  PUT  /initialize {\"start_time\":0,\"warmup_period\":86400}      → 200 · initial y")
    print("  PUT  /scenario   {\"electricity_price\":\"dynamic\"}             → 200 · time period + tariff")
    print("  PUT  /step       {\"step\":900}                                → 200 · 15-min control step")
    env = sim.RCZone(stochastic=True, seed=21)
    obs = env.reset()
    while obs[2] < 7.0:                                  # trace from 07:00 — pull-down hour
        obs, _, _, _ = env.step(sim.schedule_policy(sim.BASELINE, obs[2]))
    for _ in range(3):
        heat_sp, cool_sp = sim.schedule_policy(sim.BASELINE, obs[2])
        obs, _, _, info = env.step((heat_sp, cool_sp))
        print(f"  POST /advance    {{\"oveTSetCoo_u\":{cool_sp + 273.15:.2f},\"oveTSetCoo_activate\":1}}")
        print(f"                     → y = {{\"TZon\":{info['t_zone'] + 273.15:.2f},"
              f"\"PHVAC\":{info['power_w']:.0f}}}   (overwrite u → measurements y)")
    print("       … ×672 advances = one simulated week (point names vary per test case)")
    print("  GET  /forecast   → weather + price ahead — this endpoint is what enables MPC")
    print("  GET  /kpi        → the standard scorecard below\n")

    base_kpi = sim.boptest_kpis_for(sim.BASELINE)
    learned, _ = sim.hill_climb(sim.RCZone(stochastic=True, seed=21), episodes=24)
    rl_kpi = sim.boptest_kpis_for(learned)
    print("GET /kpi — standard KPI names, baseline controller vs our Ch 3 policy:\n")
    print(f"  {'KPI':<10}{'unit':<14}{'baseline':>10}{'Ch 3 policy':>13}   meaning")
    print("  " + "─" * 86)
    for name, unit, meaning in sim.BOPTEST_KPIS:
        b, r = base_kpi[name], rl_kpi[name]
        fmt = "{:>10.2e}{:>13.2e}" if name == "time_rat" else "{:>10.2f}{:>13.2f}"
        print(f"  {name:<10}{unit:<14}" + fmt.format(b, r) + f"   {meaning}")
    print()
    print("Why BOPTEST matters: the scenario (time period + electricity price constant/")
    print("dynamic/highly_dynamic) and the KPIs are FIXED. Your controller can't cherry-pick")
    print("its weather or its metric — that's the differentiator, and why papers cite it.\n")

    print("Honest line: on BOPTEST, well-tuned MPC generally still beats RL in the")
    print("literature — MPC exploits /forecast and a model; RL earns its place when models")
    print("are unavailable or buildings drift away from them.\n")

    view.generate("In three sentences: on the BOPTEST benchmark, when does MPC beat RL "
                  "for building control, and when is RL the right choice?",
                  max_tokens=300, title="MPC vs RL — the honest verdict")

    print("\nrun it for real (the actual service, then the same REST calls):")
    print("  $ git clone https://github.com/ibpsa/project1-boptest && cd project1-boptest")
    print("  $ TESTCASE=bestest_air docker compose up      # REST API on localhost:80")
    print("  # or `pip install boptest-gym` to drive it through the gym API of Ch 2\n")

    print("Takeaway: gyms train, BOPTEST referees. But a policy that wins in the gym is")
    print("NOT yet safe on the building — clamps, watchdogs and the safe-autonomy ladder")
    print("are App 10, where this policy earns the right to touch real setpoints.")


if __name__ == "__main__":
    main()
