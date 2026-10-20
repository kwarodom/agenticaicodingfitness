#!/usr/bin/env python3
"""PART 4 · Skills + MCP + A2A — framework-agnostic  [ADVANCED]

Write a skill once; every frontier agent loads it. Skills ride MCP for tools
(agent→tools, Week 7) and A2A for agent-to-agent delegation (agent→agent, Week 17).
This demo shows the SAME skill loaded across Claude / GPT / Nemotron, and how MCP and
A2A carry it.

Run:  python demos/step04_skills_mcp_a2a.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402


def main() -> None:
    view.banner("PART 4", "Skills + MCP + A2A — framework-agnostic", "ADVANCED")
    view.mode_line()

    print("One Skill, any agent. The SAME SKILL.md loads across every frontier agent:\n")
    for model, home, _, note in sim.CATALOG:
        print(f"  ▣ loads into {model:<24} [{home:<12}] {note}")
    print()

    print("Two open standards carry a skill — complementary, not competing:")
    print("  • MCP  (Week 7)  — agent → TOOLS.  A skill exposes its tools over MCP so any")
    print("    MCP-speaking agent can call them (nemo_retriever_search, cuopt_route, …).")
    print("  • A2A  (Week 17) — agent → AGENT.  A skill can delegate to ANOTHER agent via an")
    print("    Agent Card (/.well-known/agent.json); one agent hands a subtask to a peer.\n")

    print("So the split is:")
    print("  → MCP:  frontier agent ──tools──▶ NeMo Retriever / cuOpt / cuDF skill")
    print("  → A2A:  frontier agent ──task───▶ a specialist agent that itself loads skills\n")

    print("Because it's standardized, you avoid the N×M rebuild:")
    print("  # 🖥️ same skill, three agents — zero rewrites")
    print("  agent=claude  load-skill nemo-retriever   # or gpt / nemotron")
    print("  # tools flow over MCP; cross-agent handoff flows over A2A\n")

    view.generate("In two sentences, explain how MCP and A2A let ONE skill work across "
                  "different agents and frameworks without rewriting it.",
                  max_tokens=260, title="skills across frameworks")

    print("\nTakeaway: skills are the portable capability; MCP moves tools and A2A moves")
    print("tasks. Write once, run in any agent — the framework-agnostic way to connect a")
    print("frontier agent to your business. github.com/NVIDIA/skills")


if __name__ == "__main__":
    main()
