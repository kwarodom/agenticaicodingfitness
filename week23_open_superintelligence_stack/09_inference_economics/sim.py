#!/usr/bin/env python3
"""Inference-economics simulator — learn the cost of intelligence with no GPU.

Provides the same trio every Week 23 app ships (installed_models / tok_s /
stream_generate) plus a cost_per_mtok() helper the economics demos use to turn a
measured (or simulated) tok/s into dollars per million tokens.

All dollar figures are ILLUSTRATIVE teaching constants, not price quotes — they
exist to show the SHAPE of the economics (open-on-DGX ≈ half a cloud API), the
throughput-per-GPU / per-MW formulas, and cost-per-successful-task.
"""
from __future__ import annotations

import time

# model → (engine/where, single-stream tok/s, illustrative $ / 1M output tokens, note)
# "open on your DGX" models cost only amortized infra (≈ half a comparable cloud API);
# the cloud API line is what you'd pay a hosted frontier provider per Mtok.
CATALOG = [
    ("nemotron-3-nano:30b-a3b",    "open · your DGX",  54, 0.45, "fast sub-agents / routing"),
    ("nemotron-3-super:120b-a12b", "open · your DGX",  20, 0.90, "reasoning + tool-calling"),
    ("nemotron-rag",               "open · your DGX",  55, 0.44, "document intelligence"),
    ("llama-3.3-70b-instruct",     "open · your DGX",   6, 0.95, "general purpose"),
    ("cloud-frontier-api",         "hosted · off-box", 60, 1.80, "cloud API — usage-billed"),
]
_TOK = {m: t for m, _, t, _, _ in CATALOG}
_COST = {m: c for m, _, _, c, _ in CATALOG}

# A hosted cloud-API baseline in $ / 1M output tokens (illustrative).
CLOUD_MTOK = 1.80


def installed_models() -> list[str]:
    # the open models you'd actually have on the DGX (exclude the cloud line).
    return [m for m, *_ in CATALOG[:3]]


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40))


def cost_per_mtok(model: str) -> float:
    """Illustrative $ / 1M output tokens for a model (open-on-DGX vs cloud API)."""
    return float(_COST.get(model, 0.90))


_CANNED = ("[simulated] Tokens are the unit of AI work: cost = infra / tokens, and "
           "throughput = tokens/sec per GPU (or per megawatt). But raw tokens are cheap — "
           "only CORRECT tokens count, so measure cost per SUCCESSFUL task, which means "
           "performance and evaluation are one problem, not two.")


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)
