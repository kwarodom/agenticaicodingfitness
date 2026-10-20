#!/usr/bin/env python3
"""PART 4 · Procedural memory — the skill the agent wrote  [INTERMEDIATE]

Facts are what the agent KNOWS; skills are what it can DO. During consolidation
the agent wrote itself a reusable 'hvac-triage' skill. This demo shows that skill
and the principle that an agent improving its OWN procedures — on the DGX — is
the core of 'self-evolving'.

Run:  python demos/step04_skills.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import memory  # noqa: E402
import sevview  # noqa: E402


def main() -> None:
    sevview.banner("PART 4", "Procedural memory — the skill the agent wrote", "INTERMEDIATE")
    sevview.brain_line()

    sk = memory.skills()
    if not sk:
        print("No skills yet — run step03_consolidate.py first to let the agent write one.\n")
        return

    print(f"Skills the agent has written for itself: {', '.join(sk)}\n")
    for slug in sk:
        print(f"── skills/{slug}.md")
        for ln in memory.load_skill(slug).splitlines():
            print("    " + ln)
        print()

    print("Why procedural memory is special:")
    print("  • A skill is reusable, named know-how — the agent recalls it by name and")
    print("    follows it, instead of re-deriving the procedure every time.")
    print("  • The agent AUTHORED it from its own successful episodes — self-improvement.")
    print("  • Skills are just files on the DGX → you can review, edit, and version them")
    print("    in git, exactly like the Claude Code skills you've used all course.")
    print("  • Bad skills can be garbage-collected; good ones compound over time.")

    print("\nTakeaway: an agent that writes and reuses its own skills is evolving its")
    print("CAPABILITIES, not just its facts. Next: prove it actually got better.")


if __name__ == "__main__":
    main()
