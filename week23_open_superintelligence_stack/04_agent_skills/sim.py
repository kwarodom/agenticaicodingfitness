#!/usr/bin/env python3
"""Agent Skills simulator — learn NVIDIA Agent Skills with no GPU.

A **Skill** is packaged, portable expertise (instructions + tools + resources) that
ANY frontier agent can discover and load on demand — the standard way to connect a
general agent to YOUR business systems (github.com/NVIDIA/skills).
"""
from __future__ import annotations

import time

# The "models" here are the FRONTIER AGENTS a Skill can be loaded into. A skill is
# agent-agnostic: the same SKILL.md works across Claude, GPT, Gemini, and open Nemotron.
# (model → home, "tok/s" = illustrative streaming speed, note)
CATALOG = [
    ("claude-frontier-agent",   "Anthropic",   48, "loads skills natively (SKILL.md)"),
    ("gpt-frontier-agent",      "OpenAI",       42, "skills via MCP tools"),
    ("gemini-frontier-agent",   "Google",       40, "skills via function-calling"),
    ("nemotron-open-agent",     "NVIDIA open",  54, "sovereign, on your DGX"),
]
_TOK = {m: t for m, _, t, _ in CATALOG}

# The NVIDIA skills catalog reused across Week 23 apps (github.com/NVIDIA/skills):
# each Skill is portable expertise that connects a frontier agent to a business system.
SKILLS = [
    ("AI-Q",              "deep multi-step research over your sources"),
    ("NeMo Retriever",    "document intelligence · RAG over YOUR docs"),
    ("NeMo Evaluator",    "score & gate agent/RAG quality"),
    ("NeMo Curator",      "clean & curate training/eval data"),
    ("NeMo RL & Gym",     "reinforcement-learning environments"),
    ("NeMo Anonymizer",   "strip PII before it leaves the perimeter"),
    ("NeMo Data Designer","synthesize domain datasets"),
    ("cuOpt",             "GPU route & logistics optimization"),
    ("cuDF",              "GPU dataframe analytics on your tables"),
    ("VSS",               "video search & summarization"),
    ("Voice Chat",        "low-latency speech in/out"),
    ("TensorRT-LLM",      "optimized on-device inference"),
]


def installed_models() -> list[str]:
    """The frontier agents 'available' to load skills into (sim default set)."""
    return [m for m, *_ in CATALOG]


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 42))


_CANNED = ("[simulated skill load] Skill discovered → read SKILL.md metadata → tools "
           "loaded on demand (progressive disclosure). The frontier agent now acts on "
           "your business systems — NeMo Retriever answers from YOUR sovereign docs, no "
           "data leaves the perimeter. One Skill, any agent, via MCP + A2A. "
           "github.com/NVIDIA/skills.")


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)
