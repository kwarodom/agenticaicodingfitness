#!/usr/bin/env python3
"""PART 1 · The switchable brain  [BEGINNER]

v2's headline feature: the SAME agent + memory engine runs on a local sovereign
model (DGX/Ollama), on Claude (cloud), or on a scripted stub — flip one env var.
This proves the memory architecture is brain-agnostic, and it makes the
sovereignty trade-off explicit: only BRAIN=local keeps everything on the box.

Run:  python demos/step01_brain_switch.py        # uses whatever BRAIN is set
      BRAIN=local  python demos/step01_brain_switch.py
      BRAIN=claude python demos/step01_brain_switch.py   # needs ANTHROPIC_API_KEY
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import brain  # noqa: E402
import sevview  # noqa: E402


def main() -> None:
    sevview.banner("PART 1", "The switchable brain", "BEGINNER")
    sevview.brain_line()

    print("Set the brain with one env var — the agent code never changes:\n")
    print("  BRAIN=local    → DGX / Ollama model  (sovereign — nothing leaves the box)")
    print("  BRAIN=claude   → Anthropic Claude     (cloud — prompts DO leave the box)")
    print("  BRAIN=sim      → scripted stub         (offline, $0)")
    print("  BRAIN=auto     → local if up, else claude, else sim (the default)\n")

    print(f"Right now BRAIN={brain.name()} → {brain.where()}.\n")
    print("One thought from the active brain:\n")
    ans = brain.chat("In one sentence: why keep an agent's MEMORY on-prem, not just its model?",
                     max_tokens=160)
    for ln in ans.splitlines():
        print("    " + ln)

    print("\nWhy 'switchable' matters:")
    print("  • The memory engine (next chapters) is identical for every brain.")
    print("  • You can prototype on Claude, then go sovereign on the DGX for production —")
    print("    or vice-versa — without rewriting the agent.")
    print("  • It makes the sovereignty cost legible: 'claude' literally ships your")
    print("    prompts off-box; 'local' does not. Same code, different guarantee.")

    print("\nTakeaway: the brain is a swappable component. The rest of v2 shows the part")
    print("that makes the agent get SMARTER over time — its memory — all on the DGX.")


if __name__ == "__main__":
    main()
