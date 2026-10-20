#!/usr/bin/env python3
"""A faithful **Nemotron-family simulator** — learn the models with no GPU.

Numbers are the published Nemotron 3 specs (Nano/Super/Ultra) + representative
DGX-Spark figures. Used when no live endpoint is reachable so every chapter is
learnable offline; the real `ollama run` / `build.nvidia.com` steps are always shown.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class NemotronSpec:
    name: str          # display name
    tag: str           # an ollama-style tag you'd pull
    total_b: float     # total params (billions)
    active_b: float    # active params/token (MoE)
    ctx_k: int         # context window (thousands of tokens)
    vram_gb_nvfp4: float
    tok_s_spark: float # ~single-stream decode on one DGX Spark (GB10)
    sparks: int        # DGX Sparks needed (NVFP4)
    role: str


# The Nemotron 3 open family (from the deck: Nano 30B-A3B, Super 120B-A12B,
# Ultra 550B-A55B) + the specialized variants. Hybrid Mamba-Transformer MoE, 1M ctx.
FAMILY = [
    NemotronSpec("Nemotron 3 Nano",  "nemotron-3-nano:30b-a3b",   30,  3.0, 1000,  22, 54, 1, "fast specialized sub-agents · routing"),
    NemotronSpec("Nemotron 3 Super", "nemotron-3-super:120b-a12b",120, 12.0, 1000,  78, 20, 1, "high-accuracy reasoning + tool-calling"),
    NemotronSpec("Nemotron 3 Ultra", "nemotron-3-ultra:550b-a55b",550, 55.0, 1000, 360,  9, 2, "mission-critical multi-step reasoning"),
    NemotronSpec("Nemotron RAG",     "nemotron-rag",               8,  8.0,  128,   6, 55, 1, "document intelligence (retriever)"),
    NemotronSpec("Nemotron Speech",  "nemotron-speech",            8,  8.0,   32,   6, 48, 1, "full-duplex speech conversations"),
    NemotronSpec("Nemotron Safety",  "nemotron-safety",            8,  8.0,   16,   5, 62, 1, "content moderation / guardrails"),
]
BY_TAG = {s.tag: s for s in FAMILY}


def installed_models() -> list[str]:
    """What a Nemotron-ready DGX Spark would have pulled."""
    return ["nemotron-3-nano:30b-a3b", "nemotron-3-super:120b-a12b", "nemotron-rag"]


def spec_for(name: str) -> NemotronSpec:
    if name in BY_TAG:
        return BY_TAG[name]
    for s in FAMILY:
        if s.tag.split(":")[0] in name or s.name.lower().replace(" ", "-") in name.lower():
            return s
    return FAMILY[0]


# ── simulated generation (mechanics + reasoning shape, not real intelligence) ─
_REASON = ("<think> The user wants a concise, correct answer. I'll recall the "
           "relevant facts, check the constraint, and draft — then tighten to the "
           "requested length. </think>")
_CANNED = {
    "reason": "Running open Nemotron on your own DGX means the reasoning trace and the "
              "answer never leave the box — you get frontier-class multi-step reasoning "
              "with full data sovereignty and $0 per token.",
    "family": "Pick Nano for fast sub-agents, Super for the main reasoning/tool-calling "
              "agent, and Ultra for the hardest multi-step work — all open weights.",
    "default": "[simulated Nemotron] Open, efficient, 1M-context reasoning model built for "
               "long-running multi-agent systems — running entirely on your DGX.",
}


def _pick(prompt: str) -> str:
    p = prompt.lower()
    if any(w in p for w in ("why", "sovereign", "privacy", "reason")):
        return _CANNED["reason"]
    if any(w in p for w in ("which", "pick", "nano", "super", "ultra", "family")):
        return _CANNED["family"]
    return _CANNED["default"]


def stream_generate(prompt: str, model: str, *, show_reasoning: bool = True):
    """Yield (kind, text) chunks: 'reason' (thinking) then 'answer'."""
    rate = spec_for(model).tok_s_spark
    delay = min(0.04, 1.0 / max(rate, 1) * 1.3)
    if show_reasoning:
        for w in _REASON.split(" "):
            yield "reason", w + " "
            time.sleep(delay)
    for w in _pick(prompt).split(" "):
        yield "answer", w + " "
        time.sleep(delay)
