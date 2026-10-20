#!/usr/bin/env python3
"""Stack Navigator simulator — every layer of the stack, no GPU required.

Streams each stack node's faithful canned answer (from stack_content.json)
token-by-token at plausible DGX-Spark (GB10) tok/s, so SIM mode feels like
watching the real box think. All simulated "facts" live in the content file;
this module only paces them.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent
CONTENT = PKG / "stack_content.json"

# Plausible single-stream decode rates on a DGX Spark (GB10, 273 GB/s LPDDR5X).
# MoE models are fast (few active params); dense/big models slower.
_TOK = {
    "nemotron-3-nano:30b-a3b":    54.0,   # 30B MoE, ~3B active — snappy
    "nemotron-3-super:120b-a12b": 20.0,   # 120B MoE at NVFP4 — the workhorse
    "qwen3.6:35b-a3b-q8_0":       60.0,   # the Spark's empirically fast default
}
_DEFAULT_TOK_S = 40.0

_ANSWERS: dict[str, str] | None = None


def installed_models() -> list[str]:
    return list(_TOK)


def tok_s(model: str) -> float:
    return float(_TOK.get(model, _DEFAULT_TOK_S))


def _answers() -> dict[str, str]:
    """node_id → sim_answer, loaded once from stack_content.json."""
    global _ANSWERS
    if _ANSWERS is None:
        _ANSWERS = {}
        try:
            data = json.loads(CONTENT.read_text())
            for layer in data.get("layers", []):
                for node in layer.get("nodes", []):
                    _ANSWERS[node["id"]] = node.get("sim_answer", "")
        except Exception:
            pass
    return _ANSWERS


def answer_for(node_id: str) -> str:
    return _answers().get(node_id) or (
        "[simulated] No canned answer for this node — connect a real endpoint "
        "via the 🔌 Connection panel to run it live.")


def stream_answer(node_id: str, model: str):
    """Yield the node's sim_answer word-by-word at the model's simulated tok/s."""
    yield from stream_generate(answer_for(node_id), model)


def stream_generate(prompt_or_text: str, model: str):
    """House-convention pacer: ~1.3 tokens per word, capped at 40 ms/word."""
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in prompt_or_text.split(" "):
        yield w + " "
        time.sleep(delay)
