#!/usr/bin/env python3
"""Checkpoint 7 — Capstone: the complete self-evolving agent  (Part 10).

Assembles all three memory layers into one SelfEvolvingAgent and runs the SAME
task three times, consolidating between runs. The thesis of the whole tutorial —
COMPOUND RETURNS — becomes measurable: identical work takes fewer turns and costs
less on each run, because the agent wrote and then loaded its own SKILL.md.

    Run 1 (amnesiac)  → explore, fail, fix, verify          ← writes a skill
    Run 2 (warm)      → recall the skill, skip the discovery ← refines the skill
    Run 3 (warmer)    → fast path

Runs OFFLINE by default (deterministic simulation, $0, no network). The LIVE
version — a real LLM that genuinely improves — is the `server.py` visualizer.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from self_evolving_agent import config
from self_evolving_agent.core.agent import SelfEvolvingAgent


def main() -> None:
    print("● Checkpoint 7 — Capstone: the self-evolving agent (compound returns)\n")
    # Isolated, throwaway memory so the checkpoint is repeatable.
    tmp = Path(tempfile.mkdtemp())
    agent = SelfEvolvingAgent(db_path=str(tmp / "state.db"), memory_dir=str(tmp),
                              skills_dir=str(tmp / "skills"))
    agent.live = False                       # deterministic offline simulation
    print(f"  (live LLM mode available: {config.sdk_available()} — using offline "
          "simulation here for a deterministic, free run)\n")

    task = "Find and fix the bug in service.py that crashes on data.csv"
    runs = []
    for i in range(1, 4):
        rep = agent.memory_report(task)
        r = agent.run_task(task, label=f"run {i}")
        learned = agent.consolidate(r["session_id"])
        runs.append(r)
        skill = (learned.get("skill") or {}).get("skill")
        loaded = ", ".join(r["skills_loaded"]) or "—"
        print(f"  Run {i} │ skills loaded: {loaded:<20} │ "
              f"turns: {r['turns']} │ cost: ${r['cost']:<6} │ "
              f"consolidated→ {skill or 'facts only'}")

    print("\n  ── compound returns ──")
    t0, t_last = runs[0]["turns"], runs[-1]["turns"]
    c0, c_last = runs[0]["cost"], runs[-1]["cost"]
    print(f"  turns:  {t0} → {t_last}   ({100*(t0-t_last)//t0}% fewer)")
    print(f"  cost:   ${c0} → ${c_last}   ({100*(c0-c_last)/c0:.0f}% cheaper)")
    print(f"  skills: {[s['name'] for s in agent.skills.list_skills()]}")

    assert runs[1]["turns"] < runs[0]["turns"], "warm run should be faster"
    assert runs[0]["skills_loaded"] == [] and runs[1]["skills_loaded"], \
        "run 1 starts cold; run 2 should load the learned skill"
    agent.close()

    print("\n✓ Checkpoint 7 passed — same task, fewer turns and lower cost each run, "
          "with NO human tuning. The agent wrote the skill it then reused.")
    print("\n  ▶ Next: see the REAL LLM version evolve live —")
    print("      .venv/bin/python week18/self_evolving_agent/server.py")


if __name__ == "__main__":
    main()
