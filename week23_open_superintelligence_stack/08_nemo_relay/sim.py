#!/usr/bin/env python3
"""NeMo Relay simulator — learn observe/optimize with no GPU or endpoint.

Models the backends the Relay's **Router** picks among: a big model for hard
requests, a Nano for easy ones, a Mini in between — right-sizing per request.
"""
from __future__ import annotations

import time

# The model backends the Router/Gateway right-sizes across (name → tier, tok/s, use, $).
# Cost is illustrative relative $ per 1M output tokens (labeled, not a live price).
CATALOG = [
    ("gpt-5.5",       "big",  22,  "hard reasoning / long agent turns", 10.0),
    ("gpt-4.4-mini",  "mini", 60,  "moderate tool-use / summarize",      1.2),
    ("gpt-5.4-nano",  "nano", 120, "easy routing / classify / extract",  0.15),
]
_TOK = {m: t for m, _, t, _, _ in CATALOG}
_COST = {m: c for m, *_, c in CATALOG}


def installed_models() -> list[str]:
    return [m for m, *_ in CATALOG]


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 60))


def cost_per_mtok(model: str) -> float:
    return float(_COST.get(model, 1.0))


_CANNED = ("[simulated NeMo Relay] Observed one Hermes turn: captured the terminal + "
           "execute_code tool spans and the llm span, exported the trace to Phoenix "
           "(status ✓, cost + latency attached), and routed this request to the "
           "right-sized backend — cheaper and faster than always calling the big model.")


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)
