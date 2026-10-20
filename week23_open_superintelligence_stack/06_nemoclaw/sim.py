#!/usr/bin/env python3
"""NemoClaw simulator — build specialized agents with no GPU."""
from __future__ import annotations

import time

# The base OPEN models a NemoClaw specialized agent is built ON (model → backend → role).
# A NemoClaw spec pairs one of these with a persona + skills + tools + a signed policy.
CATALOG = [
    ("nemotron-3-nano:30b-a3b",    "TensorRT-LLM", 54, "fast, many specialists / sub-agents"),
    ("nemotron-3-super:120b-a12b", "TensorRT-LLM", 20, "deep reasoning + tool orchestration"),
    ("nemotron-rag",               "TensorRT-LLM", 55, "document-grounded specialists"),
    ("llama-3.3-70b-instruct",     "vLLM",          6, "general base for a specialist"),
    ("qwen3-32b",                  "SGLang",       14, "structured / policy-bound output"),
]
_TOK = {m: t for m, _, t, _ in CATALOG}


def installed_models() -> list[str]:
    return [m for m, *_ in CATALOG[:3]]


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40))


_CANNED = ("[simulated NemoClaw agent] I am a specialized agent: a base Nemotron model given a "
           "role/persona, a set of allowed skills and tools, and a signed policy. I run inside an "
           "OpenShell sandbox with an egress allowlist, so I can act on a task safely — nothing "
           "escapes the box. Author once, run as a domain expert.")


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)
