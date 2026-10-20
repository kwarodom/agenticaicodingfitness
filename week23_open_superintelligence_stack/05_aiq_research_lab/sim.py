#!/usr/bin/env python3
"""AI-Q simulator — learn the Open Agent Blueprint deep-research system with no GPU.

AI-Q is the HARNESS in "Agent = Model + Harness": open Nemotron models (Nano for
routing/sub-agents, Super for research) wired into a LangGraph/LangChain deep-research
multi-agent system via the NeMo Agent Toolkit. This sim serves the router/sub-agent
models so every chapter is learnable without a DGX.
"""
from __future__ import annotations

import time

# The open Nemotron models AI-Q wires into its agents (model → role in the blueprint).
CATALOG = [
    ("nemotron-3-nano:30b-a3b",    "Intent Router · shallow research · sub-agent glue", 54, "route / escalate"),
    ("nemotron-3-super:120b-a12b", "Researcher sub-agents · deep evidence gathering",   20, "research + tools"),
    ("nemotron-rag",               "NeMo Retriever RAG over your documents",            55, "document intelligence"),
    ("gpt-5.2",                    "Deep-agent orchestration (swappable · closed)",     18, "orchestration"),
    ("llama-3.3-70b-instruct",     "general-purpose fallback",                           6, "general purpose"),
]
_TOK = {m: t for m, _, t, _ in CATALOG}


def installed_models() -> list[str]:
    return [m for m, *_ in CATALOG[:3]]


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40))


_CANNED = ("[simulated AI-Q] AI-Q is the open, customizable HARNESS for deep research: "
           "an Intent Router (Nemotron Nano) sends simple questions to shallow research and "
           "escalates hard ones to a LangChain Deep Agent that plans, fans out Nemotron Super "
           "researcher sub-agents, and gathers evidence via the NeMo Agent Toolkit (web search, "
           "RAG, MCP). Best accuracy, easier to customize, more observable, ~50% lower cost with "
           "open models — nothing leaves your DGX.")


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)
