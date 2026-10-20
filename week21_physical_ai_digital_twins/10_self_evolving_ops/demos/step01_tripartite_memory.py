#!/usr/bin/env python3
"""PART 1 · Tripartite memory for building ops  [BEGINNER]

A fixed policy — even the RL-trained one from App 9 — meets a building that DRIFTS:
seasons turn, coils foul, tenants change. Week 18's tripartite memory model applied
to operations: EPISODIC control episodes, SEMANTIC building facts, PROCEDURAL control
recipes — and a background CONSOLIDATION loop that distils one into the others.

Run:  python demos/step01_tripartite_memory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

DIAGRAM = """\
              ┌───────────── the self-evolving operator's memory ─────────────┐
   control    │  EPISODIC              SEMANTIC               PROCEDURAL      │
   episodes ──┼─► what happened   ──►  durable building  ──►  control recipes │
              │  (state, action,       facts ("ballroom       (SKILL.md-style │
              │   outcome, operator     lag ≈ 45 min")         playbooks)     │
              │   reaction)                  ▲                      ▲          │
              │        └────── CONSOLIDATION┴──────────────────────┘          │
              └───────────────── (background loop, Week 18) ──────────────────┘
"""


def store_line(name: str, items: int, what: str) -> None:
    print(f"  {name:<11}{items:>3} item(s)   {what}")


def main() -> None:
    view.banner("PART 1", "Tripartite memory for building ops", "BEGINNER")
    print("▣ MODE: offline — this chapter is pure stdlib (no GPU, no endpoint, $0).\n")

    print("Why a deployed policy goes stale: the building DRIFTS. Monsoon season changes")
    print("the latent load, AHU-3's coil fouls (−20 % ΔT since March), a new tenant moves")
    print("in. A policy frozen at training time — even App 9's RL policy — slowly gets the")
    print("building wrong. The fix is Week 18's answer for coding agents, applied to ops:\n")
    print(DIAGRAM)

    print("Memory stores BEFORE — a freshly deployed agent knows nothing durable:\n")
    store_line("EPISODIC", 0, "raw control episodes")
    store_line("SEMANTIC", 0, "durable building facts")
    store_line("PROCEDURAL", 0, "control-recipe playbooks")

    print("\nReplaying 3 morning-startup episodes into EPISODIC memory:\n")
    for ep in sim.EPISODES:
        print(f"  {ep['id']} · {ep['when']}")
        print(f"    state   : {ep['state']}")
        print(f"    action  : {ep['action']}")
        print(f"    outcome : {ep['outcome']}")
        print(f"    operator: {ep['reaction']}\n")

    facts, skills = sim.consolidate(sim.EPISODES)
    print("Running CONSOLIDATION (two counting rules — see sim.consolidate):")
    print("  rule 1: an observation seen in ≥ 2 episodes  → semantic fact")
    print("  rule 2: an operator correction repeated ≥ 2× → procedural skill\n")
    print(f"  → wrote {len(facts)} semantic facts:")
    for f in facts:
        print(f"      • {f}")
    print(f"  → wrote {len(skills)} skill playbook:\n")
    for _, md in skills:
        for line in md.splitlines():
            print(f"      {line}")

    print("\nMemory stores AFTER consolidation:\n")
    store_line("EPISODIC", len(sim.EPISODES), "raw episodes (kept for the next distillation)")
    store_line("SEMANTIC", len(facts), "facts the agent recalls before every startup")
    store_line("PROCEDURAL", len(skills), "morning-startup-monsoon-season playbook")

    print("\nTakeaway: episodes are cheap and noisy; consolidation turns them into facts and")
    print("skills the agent reuses forever — the exact loop Week 18 used to make a coding")
    print("agent measurably cheaper run after run. Next: overrides as free training data.")


if __name__ == "__main__":
    main()
