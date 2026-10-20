#!/usr/bin/env python3
"""PART 4 · Researcher sub-agents fan out  [INTERMEDIATE]

The orchestrator dispatches the plan's independent sub-tasks to parallel Researcher
Sub-Agents — each a Nemotron 3 Super instance that calls tools (web search, NeMo Retriever
RAG) to gather evidence and returns findings to Memory. Fan-out is the multi-agent pattern:
many focused Super agents concurrently beat one model holding the whole task in context.
This demo prints the dispatch/return trace and, in REAL mode, runs one researcher for real.

Run:  python demos/step03_researchers_fanout.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import view  # noqa: E402


def main() -> None:
    view.banner("PART 4", "Researcher sub-agents fan out", "INTERMEDIATE")
    view.mode_line()

    print("Orchestrator → N parallel Nemotron Super researchers, each with its own tools:\n")
    print("  • Researcher A → Tavily web search      • Researcher B → NeMo Retriever RAG")
    print("  • each returns findings to filesystem/Memory; orchestrator synthesizes\n")

    if view.is_sim():
        print("SIM — the fan-out trace the orchestrator runs (REAL mode calls the model):\n")
        print("  → DISPATCH Researcher A (Super): 'DGX 3-yr hardware + power TCO'")
        print("  → DISPATCH Researcher B (Super): 'frontier API $/1M tok + team volume'")
        print("    · A → ACT web_search('DGX Spark price power draw depreciation')")
        print("    · A ← OBSERVE 3 sources · capex + ~kWh/yr extracted")
        print("    · B → ACT rag_retrieve('internal: analyst monthly token usage')")
        print("    · B ← OBSERVE 2 doc chunks · ~9.4M tok/analyst/month")
        print("  ← Researcher A findings → Memory   ← Researcher B findings → Memory")
        print("  ⇒ orchestrator synthesizes both into a cited comparison")
        print("\n  ◆ 2 researchers in parallel · Super for depth, Nano routed the request · $0")
    else:
        view.generate(
            "You are one Researcher Sub-Agent (Nemotron Super) in a deep-research fan-out. Your "
            "assigned sub-task: estimate the 3-year hardware + power total cost of owning an "
            "NVIDIA DGX for a small research team. In 3 sentences, state what you'd search for, "
            "the tools you'd call, and the evidence you'd return to Memory.",
            max_tokens=300, title="one researcher sub-agent works its task")

    print("\nTakeaway: parallel focused Super agents + tools = deep coverage without one model")
    print("drowning in context. Findings land in shared Memory. Next: the tool bus itself.")


if __name__ == "__main__":
    main()
