#!/usr/bin/env python3
"""PART 2 · Episodic memory — log what happened  [BEGINNER]

The first memory store is EPISODIC: an append-only log of every task and answer,
written to .memory/episodes.jsonl ON THE DGX. Nothing is consolidated yet — this
is raw experience. We run two HVAC tasks and watch the log grow.

Run:  python demos/step02_episodic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import memory  # noqa: E402
import sevview  # noqa: E402

TASKS = [
    "Room 1203 is occupied and reads 26.4 °C (setpoint 22). Triage and act.",
    "Room 0907 is empty; filter ΔP is 265 Pa. Triage and act.",
]


def main() -> None:
    sevview.banner("PART 2", "Episodic memory — log what happened", "BEGINNER")
    sevview.brain_line()

    print("Running two tasks; each is logged to the on-DGX episodic store.\n")
    for i, task in enumerate(TASKS, 1):
        print(f"── Task {i}")
        sevview.run_task(task, session="cp2", show_recall=(i > 1))

    eps = memory.episodes()
    print(f"Episodic log now holds {len(eps)} events (.memory/episodes.jsonl):")
    for e in eps[-4:]:
        print(f"  [{e['kind']:<6}] {e['content'][:70]}")
    print()

    print("Properties of episodic memory:")
    print("  • Append-only + timestamped → a faithful, auditable history of behaviour.")
    print("  • It's raw: lots of detail, no generalization yet (that's consolidation).")
    print("  • It lives on the DGX — this log is your operational record, not a vendor's.")

    print("\nTakeaway: the agent now REMEMBERS what it did. Next: distill that raw")
    print("experience into durable facts + skills — the 'sleep' loop.")


if __name__ == "__main__":
    main()
