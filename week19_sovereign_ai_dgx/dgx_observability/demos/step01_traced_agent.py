#!/usr/bin/env python3
"""PART 1 · Trace a sovereign agent  [BEGINNER]

You can't manage what you can't see. This runs a small smart-hotel HVAC triage
agent against the DGX model and wraps every step — the agent, each LLM call, each
tool call — in an OpenTelemetry-shaped span. The result is the exact span tree
Arize Phoenix would show you, rendered right here.

Run:  python demos/step01_traced_agent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import obsview  # noqa: E402

TASK = ("Clear the morning HVAC alarm queue. Room 1203 (occupied) is at 26.4 °C; "
        "room 0907 (empty) has filter ΔP 265 Pa. Triage and dispatch correctly.")


def main() -> None:
    obsview.banner("PART 1", "Trace a sovereign agent", "BEGINNER")
    obsview.mode_line()

    print("Task given to the agent:")
    print(f"  {TASK}\n")
    print("Running the agent — each step is captured as a span…\n")

    tr = obsview.traced_agent_run(TASK)
    obsview.show_tree(tr)

    print("\nWhat you're looking at (the Phoenix mental model):")
    print("  • A TRACE is one agent run; SPANS are the steps, nested by parent.")
    print("  • Span KINDS (AGENT / LLM / TOOL) let Phoenix group + filter.")
    print("  • Each span carries latency, token usage, and input/output values.")
    print("  • The waterfall shows WHERE the time + tokens went — your first")
    print("    debugging tool when an agent is slow, wrong, or expensive.")

    print("\nTakeaway: a sovereign agent isn't a black box — instrument it once and")
    print("every run becomes an inspectable trace. Next: pipe these into real Phoenix.")


if __name__ == "__main__":
    main()
