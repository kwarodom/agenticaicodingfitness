#!/usr/bin/env python3
"""PART 3 · Deep Agent — orchestrate & plan  [INTERMEDIATE]

When the router escalates, the LangChain Deep Agent takes over. Its Orchestration layer
(GPT 5.2 or an open Nemotron, wired via NAT) hands the task to a Planning Sub-Agent that
decomposes it into an ordered research plan written to the filesystem To-Do. Decompose
first, delegate second — a written plan is what makes a long run observable and resumable.
This demo shows the planning prompt and, in REAL mode, asks the model to produce the plan.

Run:  python demos/step02_deep_agent_plan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import view  # noqa: E402

TASK = ("Compare the 3-year total cost of ownership of running Nemotron on a DGX vs a "
        "frontier API for a 10-analyst research team.")

STRUCTURE = """\
# LangChain Deep Agent (escalated from the router)
#   Orchestration      — GPT 5.2 / Nemotron, wired via NAT   (owns the run)
#     └─ Planning Sub-Agent  — decompose task → ordered plan → filesystem To-Do
#          └─ Researcher Sub-Agents  — execute plan items in parallel (next chapter)
"""


def main() -> None:
    view.banner("PART 3", "Deep Agent — orchestrate & plan", "INTERMEDIATE")
    view.mode_line()

    print("The Deep Agent decomposes BEFORE it researches — plan to file, then delegate:\n")
    print(STRUCTURE)
    print(f'» escalated task: "{TASK}"\n')

    if view.is_sim():
        print("SIM — the plan the Planning Sub-Agent writes to To-Do (REAL mode generates it):\n")
        plan = [
            "define TCO scope: hardware capex, power, staff, model/API fees, support",
            "gather DGX Spark/Station price, power draw, depreciation over 3 years",
            "gather frontier-API $/1M tokens + estimate a 10-analyst team's monthly volume",
            "compute + compare both 3-year totals; note break-even token volume",
            "synthesize: recommendation with assumptions + sensitivity",
        ]
        for i, item in enumerate(plan, 1):
            dep = "" if i == 1 else f"  (after step {i-1})"
            print(f"  → TODO {i}. {item}{dep}")
        print("\n  ← plan written to filesystem/To-Do · 5 tasks · 3 dispatchable in parallel")
        print("  ~ reason: steps 2 & 3 are independent → fan out to two Super researchers next")
    else:
        view.generate(
            "You are the Planning Sub-Agent of a deep-research agent. Decompose this task into "
            "a numbered plan of 4-6 concrete research steps (one line each), noting which steps "
            f"can run in parallel. Task: {TASK}",
            max_tokens=350, title="planning sub-agent decomposes the task")

    print("\nTakeaway: an explicit written plan is the difference between an observable,")
    print("resumable research run and one giant opaque prompt. Next: fan out the researchers.")


if __name__ == "__main__":
    main()
