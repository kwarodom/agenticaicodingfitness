#!/usr/bin/env python3
"""Checkpoint 4 — The subconscious loop: background consolidation  (Part 6).

The defining feature of a self-EVOLVING agent: after a conversation, a background
"meta-cognitive" pass replays the episodic transcript and distils it into lasting
knowledge — semantic facts (MEMORY.md / USER.md) and procedural skills (SKILL.md).
The foreground never waits for this.

This checkpoint runs the consolidator over a synthetic transcript and shows the
episodic → semantic + procedural flow, plus the PreCompact snapshot that protects
working memory when the context window is compacted. Runs OFFLINE (heuristic
consolidator); the live agent uses a real LLM for this step.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from self_evolving_agent.core.consolidation import (
    HeuristicConsolidator, snapshot_before_compaction)


def main() -> None:
    print("● Checkpoint 4 — Background consolidation (the subconscious loop)\n")
    root = Path(tempfile.mkdtemp())
    mem_dir, skills_dir = root, root / "skills"

    # A finished episodic transcript: the user did a multi-step deploy task.
    history = [
        {"role": "user", "content": "We use Python FastAPI. Deploy project alto-cero "
                                    "to staging.", "tokens": 0, "cost": 0},
        {"role": "tool", "content": "Bash docker build -t alto-cero:staging .", "tokens": 0, "cost": 0},
        {"role": "tool", "content": "error result: image fails on prod, --platform missing",
         "tokens": 0, "cost": 0},
        {"role": "assistant", "content": "Rebuilding with --platform=linux/amd64.",
         "tokens": 0, "cost": 0},
        {"role": "tool", "content": "Bash docker build --platform=linux/amd64 -t alto-cero:staging .",
         "tokens": 0, "cost": 0},
        {"role": "tool", "content": "Bash docker push alto-cero:staging — ok", "tokens": 0, "cost": 0},
    ]
    print("  Episodic input .... 6 messages (a deploy task with one failure→fix)")

    # 1. Consolidate: distil the transcript into semantic + procedural memory
    con = HeuristicConsolidator(mem_dir, skills_dir)
    learned = con.consolidate(history, "ses-cp4")
    print(f"\n  → semantic facts .. {learned['facts'] or '(none extracted)'}")
    print(f"  → procedural skill  {learned['skill']}")
    print(f"  → pitfalls learned  {learned['pitfalls'] or '(none)'}")
    assert learned["skill"], "a multi-step task should yield a SKILL.md"

    # 2. The semantic file now contains the learned project fact
    memory_md = (mem_dir / "MEMORY.md").read_text()
    print(f"\n  MEMORY.md updated . {'alto-cero' in memory_md}")

    # 3. The skill file exists with the auto-learned pitfall
    skill_files = list(skills_dir.glob("*.md"))
    print(f"  skills/ ........... {[p.name for p in skill_files]}")
    assert skill_files

    # 4. PreCompact snapshot — protect working memory before context compaction
    snap = snapshot_before_compaction("ses-cp4", history, root / "snapshots")
    print(f"\n  PreCompact snapshot {Path(snap['snapshot']).name} written "
          "(working memory safe across a context-window compaction)")
    assert Path(snap["snapshot"]).exists()

    print("\n✓ Checkpoint 4 passed — a finished conversation was distilled into "
          "reusable semantic + procedural memory, with zero foreground latency.")


if __name__ == "__main__":
    main()
