#!/usr/bin/env python3
"""Capstone simulator — the two Nemotron tiers the hotel runs on, no GPU needed."""
from __future__ import annotations

import time

# model → (tok/s) the NIM would serve on one DGX Spark
CATALOG = [
    ("nemotron-3-super:120b-a12b", 20, "orchestration + hard triage reasoning"),
    ("nemotron-3-nano:30b-a3b", 54, "cheap sub-agents: routing, energy, guest"),
]
_TOK = {m: t for m, t, _ in CATALOG}


def installed_models() -> list[str]:
    return [m for m, *_ in CATALOG]


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40))


_CANNED = ("[simulated Nemotron] The autonomous hotel runs on your DGX: Nano handles cheap "
           "sub-agents (routing, energy, guest), Super orchestrates and does hard triage — "
           "policy-gated, observed, self-improving. Nothing leaves the box.")


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)
