#!/usr/bin/env python3
"""PART 2 · Intent Router — route or escalate  [BEGINNER]

The front door of AI-Q. A cheap, fast Nemotron 3 Nano (on LangGraph, tuned by the NAT
Optimizer) reads a query and decides: send it to Shallow Research, or ESCALATE it to the
LangChain Deep Agent. Routing cheaply is how AI-Q hits ~50% lower cost — you don't wake the
big agent to answer "who is X?". This demo shows the routing prompt and, in REAL mode, asks
the connected Nemotron endpoint to make the call itself.

Run:  python demos/step01_intent_router.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import view  # noqa: E402

ROUTER = """\
# the AI-Q Intent Router (LangGraph node, Nemotron 3 Nano, NAT Optimizer)
#   in:  the user's query
#   out: {"route": "shallow" | "deep", "why": "..."}
#
#   shallow → Shallow Research  (Nano + NAT + Dynamo priority routing) — one lookup, seconds
#   deep    → escalate to the LangChain Deep Agent (plan → fan out Super researchers) — minutes
"""

QUERIES = [
    ("Who is the current CEO of NVIDIA?", "shallow",
     "single fact, one source — no planning needed"),
    ("Compare the total cost of ownership of running Nemotron on a DGX vs a "
     "frontier API for a 10-analyst research team over 3 years.", "deep",
     "multi-step, multi-source, needs a plan + parallel research → escalate"),
]


def main() -> None:
    view.banner("PART 2", "Intent Router — route or escalate", "BEGINNER")
    view.mode_line()

    print("AI-Q's front door: a cheap Nano classifies every query before any big agent runs.\n")
    print(ROUTER)

    if view.is_sim():
        print("SIM — the routing decision Nano makes (REAL mode asks the model):\n")
        for q, route, why in QUERIES:
            arrow = "Shallow Research" if route == "shallow" else "ESCALATE → Deep Agent"
            print(f'  » query: "{q[:66]}{"…" if len(q) > 66 else ""}"')
            print(f"  ~ reason: {why}")
            print(f"  → route: {route.upper():7s} → {arrow}\n")
        print("  ◆ ~2 of 3 queries stay shallow · Nano is ~9x cheaper than the deep agent · $0")
    else:
        hits = 0
        for q, route, _why in QUERIES:
            got = view.classify(
                "You are the AI-Q Intent Router. Classify this query: answer 'shallow' for a "
                "simple single-source lookup, or 'deep' for a multi-step, multi-source research "
                f"question that needs a plan. Query: {q}",
                labels=["shallow", "deep"], title=f"routing: {q[:48]}…")
            mark = "✓" if got == route else ("✗" if got else "—")
            hits += got == route
            print(f"  {mark} expected {route.upper()}, model said {str(got or 'no answer').upper()}\n")
        print(f"  ◆ router accuracy on this pair: {hits}/{len(QUERIES)} — in production the "
              f"NAT Optimizer tunes this prompt against a golden set (Week 10/15).")

    print("\nTakeaway: routing before reasoning is the cost lever. Cheap Nano triages;")
    print("only genuinely hard questions escalate to the expensive Deep Agent. Next: the plan.")


if __name__ == "__main__":
    main()
