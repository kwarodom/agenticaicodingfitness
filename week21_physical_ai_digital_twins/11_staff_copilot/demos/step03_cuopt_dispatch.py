#!/usr/bin/env python3
"""PART 3 · cuOpt dispatch — who goes where, in what order  [ADVANCED]

NVIDIA cuOpt is GPU-accelerated optimization: vehicle routing (VRP/TSP with time
windows, capacities, priorities) plus LP/MILP. Open-sourced at GTC March 2025
(Apache-2.0, github.com/NVIDIA/cuopt), also served as a NIM microservice. In a
building portfolio it routes technicians, cleaning robots and deliveries. This demo
solves 8 work orders × 3 technicians with a pure-Python greedy + 2-opt stand-in —
honestly labeled: it teaches the PROBLEM SHAPE, not cuOpt's solver quality.

Run:  python demos/step03_cuopt_dispatch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

WO = {w[0]: w for w in sim.DISPATCH_WOS}


def route_stats(tech: str, route: list[str]):
    """Walk a route from the tech's start at 08:00: (travel_min, wait_min, end_min, late)."""
    pos, t, travel, wait, late = sim.TECH_START[tech], 0, 0, 0, 0
    for wid in route:
        _, _, _, site, svc, (lo, hi) = WO[wid]
        leg = sim.travel_min(pos, site)
        travel, t = travel + leg, t + leg
        if t < lo:
            wait, t = wait + (lo - t), lo
        if t > hi:
            late += 1
        t, pos = t + svc, site
    return travel, wait, t, late


def two_opt(tech: str, route: list[str]) -> list[str]:
    """Classic 2-opt: reverse segments while it shortens travel + wait."""
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if sum(route_stats(tech, cand)[:2]) < sum(route_stats(tech, best)[:2]):
                    best, improved = cand, True
    return best


def show(label: str, routes: dict[str, list[str]]):
    tot_travel = tot_wait = tot_late = 0
    print(f"  {label}")
    for tech, route in routes.items():
        travel, wait, end, late = route_stats(tech, route)
        tot_travel, tot_wait, tot_late = tot_travel + travel, tot_wait + wait, tot_late + late
        stops = " → ".join(route) or "(idle)"
        print(f"    {tech:<8} {stops}")
        print(f"    {'':<8} travel {travel} min · wait {wait} min · done "
              f"{8 + end // 60:02d}:{end % 60:02d} · late stops {late}")
    print(f"    TOTAL    travel {tot_travel} min · wait {tot_wait} min · late {tot_late}\n")
    return tot_travel, tot_wait, tot_late


def main() -> None:
    view.banner("PART 3", "cuOpt dispatch — who goes where, in what order", "ADVANCED")
    try:
        import cuopt  # noqa: F401
        print("▣ MODE: REAL — cuopt is importable; this demo still runs the tiny pure-Python")
        print("  stand-in (the real command block below is the cuOpt path).\n")
    except Exception:
        print("▣ MODE: SIM — pure-Python greedy + 2-opt stands in for cuOpt ($0, no GPU).")
        print("  Honest label: same problem shape, none of cuOpt's solver quality/scale.\n")

    print("The morning board — 8 work orders, 3 technicians, skills + time windows:\n")
    print(f"  {'WO':<9}{'task':<44}{'skill':<12}{'window (after 08:00)'}")
    print("  " + "─" * 84)
    for wid, task, skill, _, svc, (lo, hi) in sim.DISPATCH_WOS:
        print(f"  {wid:<9}{task:<44}{skill:<12}{lo}–{hi} min · {svc} min job")
    for tech, sk in sim.TECH_SKILLS.items():
        print(f"  {tech:<9}skills: {', '.join(sorted(sk))} · starts at grid {sim.TECH_START[tech]}")
    print()

    # NAIVE: ticket order, first technician holding the skill. No geography, no windows.
    naive = {t: [] for t in sim.TECH_SKILLS}
    for wid, _, skill, *_ in sim.DISPATCH_WOS:
        pick = next(t for t in sim.TECH_SKILLS if skill in sim.TECH_SKILLS[t])
        naive[pick].append(wid)
    n_travel, n_wait, n_late = show("NAIVE — ticket order, first skill match:", naive)

    # OPTIMIZED (stand-in): greedy earliest-finish insertion, then 2-opt per route.
    opt = {t: [] for t in sim.TECH_SKILLS}
    clock = {t: (sim.TECH_START[t], 0) for t in sim.TECH_SKILLS}   # (pos, minutes)
    todo = [w[0] for w in sim.DISPATCH_WOS]
    while todo:
        best = None
        for wid in todo:
            _, _, skill, site, svc, (lo, hi) = WO[wid]
            for tech, (pos, t) in clock.items():
                if skill not in sim.TECH_SKILLS[tech]:
                    continue
                fin = max(t + sim.travel_min(pos, site), lo) + svc
                if best is None or fin < best[0]:
                    best = (fin, tech, wid, site)
        fin, tech, wid, site = best
        opt[tech].append(wid)
        clock[tech] = (site, fin)
        todo.remove(wid)
    opt = {t: two_opt(t, r) for t, r in opt.items()}
    o_travel, o_wait, o_late = show("OPTIMIZED — greedy + 2-opt (the cuOpt stand-in):", opt)

    print("Minutes saved:")
    print(f"  {'metric':<16}{'naive':>8}{'optimized':>11}{'saved':>8}")
    print("  " + "─" * 44)
    for name, a, b in [("travel min", n_travel, o_travel), ("wait min", n_wait, o_wait),
                       ("late stops", n_late, o_late)]:
        print(f"  {name:<16}{a:>8}{b:>11}{a - b:>8}")
    print(f"\n  {n_travel + n_wait - o_travel - o_wait} technician-minutes back per morning — "
          "that's a whole extra work order.\n")

    print("# run it for real (the actual solver, on a GPU):")
    print("#   pip install cuopt-server cuopt-sh-client   # open-source, Apache-2.0 (GTC Mar 2025)")
    print("#   docker run --gpus all nvcr.io/nim/nvidia/cuopt:latest   # or as a NIM microservice")
    print("#   github.com/NVIDIA/cuopt — VRP/TSP with windows, capacities, priorities + LP/MILP\n")

    print("Takeaway: dispatch is a solved OR problem — feed it the twin's locations and the")
    print("CMMS's windows and stop routing by gut. Next: giving the twin EYES — Metropolis.")


if __name__ == "__main__":
    main()
