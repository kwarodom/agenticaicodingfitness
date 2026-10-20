#!/usr/bin/env python3
"""PART 5 · Tools & data via the NeMo Agent Toolkit  [ADVANCED]

How AI-Q's agents touch the world. The NeMo Agent Toolkit (Week 16) is the tool bus that
connects Documents, Tavily web search, NeMo Retriever RAG, MCP servers, and the AI Data
Platform — plus Sandbox skills (Data Analysis, Image Processing) and a filesystem for
To-Do / Memory / Files. Swap in your own tools + data and you have a custom research lab.
This demo shows the toolkit wiring and, in REAL mode, has the model pick tools for a task.

Run:  python demos/step04_tools_toolkit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import view  # noqa: E402

TOOLKIT = """\
# NeMo Agent Toolkit — the tool bus every AI-Q sub-agent shares (Week 16)
tools:
  - documents           # your uploaded PDFs / files
  - web_search          # Tavily
  - rag                 # NeMo Retriever RAG over your corpus
  - mcp                 # any MCP server (databases, SaaS, internal APIs)
  - ai_data_platform    # enterprise data
sandbox_skills:
  - data_analysis       # run code on retrieved data
  - image_processing    # charts, diagrams, vision
filesystem: [ to_do, memory, files ]   # the agent's scratch space + working memory
"""


def main() -> None:
    view.banner("PART 5", "Tools & data via the NeMo Agent Toolkit", "ADVANCED")
    view.mode_line()

    print("The toolkit is what makes AI-Q a lab for YOUR domain — swap tools + data, keep the agents:\n")
    print(TOOLKIT)

    if view.is_sim():
        print("SIM — how a researcher selects from the tool bus for one sub-task:\n")
        print("  » sub-task: 'find + chart our team's monthly token spend vs the DGX break-even'")
        print("  → ACT rag('internal finance: monthly token usage 2024-2026')")
        print("  ← OBSERVE 6 chunks → filesystem/Files")
        print("  → ACT data_analysis(sandbox): compute monthly $ and break-even crossover")
        print("  → ACT image_processing(sandbox): render a line chart of the crossover")
        print("  ← OBSERVE chart.png + table → filesystem/Memory")
        print("  ⇒ every tool call is logged → observable, auditable, replayable")
        print("\n  ◆ 3 toolkit calls · Documents+RAG+Sandbox · nothing left your DGX · $0")
    else:
        view.generate(
            "You are an AI-Q researcher with these tools: documents, web_search (Tavily), rag "
            "(NeMo Retriever), mcp, ai_data_platform, and sandbox skills data_analysis + "
            "image_processing. For the task 'chart our team's monthly token spend against the "
            "DGX break-even point', list which tools you'd call, in order, and why (one line each).",
            max_tokens=300, title="picking tools from the NeMo Agent Toolkit")

    print("\nTakeaway: the NeMo Agent Toolkit is the pluggable tool + data layer. Point it at your")
    print("documents, RAG corpus, and MCP servers and AI-Q becomes a research lab for any domain.")


if __name__ == "__main__":
    main()
