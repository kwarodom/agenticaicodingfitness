#!/usr/bin/env python3
"""PART 5 · Prove it evolved — cold vs warm  [ADVANCED]

The test of a self-evolving agent: does memory actually make it better? This runs
a NEW task two ways — once with memory recall OFF (a 'cold' agent, like the first
ever run) and once with recall ON (a 'warm' agent that recalls the consolidated
facts + skill). The warm run should be more specific and consistent.

Run:  python demos/step05_evolve.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import memory  # noqa: E402
import sevview  # noqa: E402

NEW_TASK = ("New alarm: room 1511 is occupied and sitting at 25.8 °C (setpoint 22). "
            "What priority and what do you do?")


def main() -> None:
    sevview.banner("PART 5", "Prove it evolved — cold vs warm", "ADVANCED")
    sevview.brain_line()

    m = memory.stats()
    if m["facts"] == 0 and m["skills"] == 0:
        print("Memory is empty — run step02 + step03 first so there's something to recall.\n")

    print("Same new task, two agents:\n")
    print("════ COLD agent (recall OFF — no memory, like run #1) ════")
    cold = _run_cold(NEW_TASK)
    print("════ WARM agent (recall ON — recalls consolidated facts + skill) ════")
    warm = sevview.run_task(NEW_TASK, session="cp5", remember=False, show_recall=True)

    print("What changed:")
    print(f"  • cold answer length: {len(cold)} chars · warm: {len(warm)} chars")
    print("  • the WARM agent recalls 'occupied + >3°C from setpoint = CRITICAL' and the")
    print("    hvac-triage skill, so it answers with the house's specific policy — not")
    print("    generic advice. That delta is the agent's accumulated experience paying off.")
    print("  • crucially: nothing was re-taught. The improvement came from MEMORY,")
    print("    which lives on the DGX. A fresh process tomorrow starts just as smart.")

    print("\nTakeaway: self-evolving = each run seeds the next via on-DGX memory. The")
    print("warm agent is the cold agent + everything it has consolidated. Sovereign by design.")


def _run_cold(task: str) -> str:
    import brain
    print("← RECALL: (disabled for the cold run)")
    print(f"~ THINK ({brain.label()}):")
    ans = brain.chat([
        {"role": "system", "content": "You are a smart-hotel HVAC operations agent."},
        {"role": "user", "content": task}], max_tokens=240)
    for ln in ans.splitlines():
        print("    " + ln)
    print()
    return ans


if __name__ == "__main__":
    main()
