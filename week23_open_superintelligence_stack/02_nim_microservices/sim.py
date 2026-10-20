#!/usr/bin/env python3
"""NIM simulator — learn NVIDIA Inference Microservices with no GPU."""
from __future__ import annotations

import time

# A slice of the build.nvidia.com NIM catalog (model → backend the NIM auto-selects).
CATALOG = [
    ("nemotron-3-super:120b-a12b", "TensorRT-LLM", 20, "reasoning + tool-calling"),
    ("nemotron-3-nano:30b-a3b",    "TensorRT-LLM", 54, "fast sub-agents / routing"),
    ("nemotron-rag",               "TensorRT-LLM", 55, "document intelligence"),
    ("llama-3.3-70b-instruct",     "vLLM",          6, "general purpose"),
    ("qwen3-32b",                  "SGLang",       14, "structured output"),
]
_TOK = {m: t for m, _, t, _ in CATALOG}


def installed_models() -> list[str]:
    return [m for m, *_ in CATALOG[:3]]


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40))


_CANNED = ("[simulated NIM] One container — model + an auto-selected optimized engine "
           "(TensorRT-LLM/vLLM/SGLang) + an OpenAI-compatible API — running on your DGX. "
           "Deploy in one command; nothing leaves the box.")


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)
