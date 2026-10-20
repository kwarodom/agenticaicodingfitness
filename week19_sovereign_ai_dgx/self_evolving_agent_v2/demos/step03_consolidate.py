#!/usr/bin/env python3
"""PART 3 · Consolidation — the 'sleep' loop  [INTERMEDIATE]

Raw episodes are noisy. CONSOLIDATION is the subconscious step: the brain reads
recent episodes and distills them into durable FACTS (semantic memory → MEMORY.md)
and a reusable SKILL (procedural memory → skills/). This is how the agent turns
experience into reusable knowledge — and it all happens on the DGX.

Run:  python demos/step03_consolidate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import memory  # noqa: E402
import sevview  # noqa: E402


def main() -> None:
    sevview.banner("PART 3", "Consolidation — the 'sleep' loop", "INTERMEDIATE")
    sevview.brain_line()

    if not memory.episodes():
        print("No episodes yet — seeding a couple so consolidation has something to chew on.")
        memory.log_episode("task", "Occupied room 1203 at 26.4°C (setpoint 22) — dispatched CRITICAL.")
        memory.log_episode("task", "Empty room 0907, filter ΔP 265 Pa — dispatched ROUTINE.")
        print()

    print(f"Consolidating {len(memory.episodes())} episodes with the {sevview.__name__}'s brain…\n")
    result = memory.consolidate()

    print("After consolidation:")
    print(f"  • +{result['added_facts']} new fact(s) → semantic memory (MEMORY.md)")
    if result["added_skill"]:
        print(f"  • +1 skill written → procedural memory (skills/{result['added_skill']}.md)")
    print()

    print("Semantic memory now (durable facts the agent will recall forever):")
    for f in memory.facts():
        print(f"  - {f}")
    print()
    if memory.skills():
        print(f"Procedural memory (skills): {', '.join(memory.skills())}")
        print()

    print("The biological analogy (from Week 18):")
    print("  • Episodic = today's events (hippocampus); cheap to write, noisy.")
    print("  • Consolidation during 'sleep' moves the durable parts to long-term memory.")
    print("  • Semantic = facts you just KNOW; Procedural = skills you can DO.")

    print("\nTakeaway: the agent compressed raw experience into knowledge it can reuse.")
    print("Next: inspect the skill it wrote for itself.")


if __name__ == "__main__":
    main()
