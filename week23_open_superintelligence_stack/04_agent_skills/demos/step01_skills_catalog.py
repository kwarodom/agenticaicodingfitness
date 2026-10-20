#!/usr/bin/env python3
"""PART 1 · The skills catalog — portable expertise for any agent  [BEGINNER]

A **Skill** is packaged expertise (instructions + tools + resources) a frontier agent
can discover and load on demand. NVIDIA ships an open catalog of them at
github.com/NVIDIA/skills — each connects a general agent to a business capability.
This demo lists the catalog and what each skill connects.

Run:  python demos/step01_skills_catalog.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402


def main() -> None:
    view.banner("PART 1", "The skills catalog — portable expertise for any agent", "BEGINNER")
    view.mode_line()

    print("A Skill = instructions + tools + resources, packaged so ANY frontier agent")
    print("(Claude, GPT, Gemini, open Nemotron) can DISCOVER and LOAD it on demand.\n")

    print("The NVIDIA skills catalog (github.com/NVIDIA/skills) — reused across Week 23 apps:")
    for name, connects in sim.SKILLS:
        print(f"  ▣ {name:<18} → {connects}")
    print()

    print("Frontier agents these skills can load into (all use the SAME SKILL.md):")
    for model, home, _, note in sim.CATALOG:
        print(f"  · {model:<24} [{home:<12}] {note}")
    print()

    print("Grab a skill in one command:")
    print("  # 🖥️ get the open skills catalog")
    print("  git clone https://github.com/NVIDIA/skills")
    print("  ls skills/    #  → each dir is a portable Skill (SKILL.md + tools/)\n")

    view.generate("In one sentence, why is a portable Skill catalog better than hard-coding "
                  "each capability into every agent?", max_tokens=200,
                  title="why a skills catalog")

    print("\nTakeaway: skills are reusable capabilities you pick off a shelf — no rebuilding")
    print("the agent. Next: how a frontier agent actually LOADS one.")


if __name__ == "__main__":
    main()
