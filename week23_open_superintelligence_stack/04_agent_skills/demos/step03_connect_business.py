#!/usr/bin/env python3
"""PART 3 · Connect to your business — a skill over YOUR data  [INTERMEDIATE]

A skill wrapping NeMo Retriever RAG over YOUR document store turns a general frontier
agent into an expert on your business — and the data never leaves the perimeter.
Query → retrieve grounded chunks → answer. This demo runs the grounded loop and, in
REAL mode, has the endpoint compose the sovereign answer.

Run:  python demos/step03_connect_business.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import view  # noqa: E402

# YOUR sovereign document store — stays on your box; the skill retrieves from it.
DOCSTORE = {
    "sla": "Enterprise SLA: 99.9% uptime; 1-hour response for Sev-1 (runbook OPS-7).",
    "onboarding": "New-tenant onboarding takes 3 business days incl. SSO setup (DOC-42).",
}


def retrieve(query: str) -> list[str]:
    q = query.lower()
    return [v for k, v in DOCSTORE.items() if k in q] or ["(no chunk matched — do not guess)"]


def main() -> None:
    view.banner("PART 3", "Connect to your business — a skill over YOUR data", "INTERMEDIATE")
    view.mode_line()

    print("The NeMo Retriever skill wraps RAG over YOUR docs. The frontier agent gets your")
    print("expertise; your documents stay sovereign — nothing leaves the perimeter.\n")

    question = "What is our enterprise SLA?"
    chunks = retrieve("sla")
    print(f"  » QUERY (from the agent): {question}")
    print("  → skill calls nemo_retriever_search over YOUR docstore")
    for ch in chunks:
        print(f"  ← RETRIEVE chunk: {ch}")
    print()

    grounded = ("Answer the question using ONLY these retrieved chunks; cite the doc id.\n\n"
                f"Chunks:\n- " + "\n- ".join(chunks) + f"\n\nQuestion: {question}")
    print("The agent now answers, grounded in your sovereign data:\n")
    view.generate(grounded, max_tokens=220, title="grounded answer from YOUR docs")

    print("\n▣ SOVEREIGN ✓ — retrieval + generation ran against your systems; data stayed put.")
    print("Takeaway: one skill connected a general agent to YOUR business. Next: the same")
    print("skill, reused across agents & frameworks via MCP + A2A.")


if __name__ == "__main__":
    main()
