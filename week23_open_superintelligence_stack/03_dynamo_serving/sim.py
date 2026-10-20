#!/usr/bin/env python3
"""NVIDIA Dynamo simulator — learn distributed serving with no GPU."""
from __future__ import annotations

import time

# Serving profiles: naive single-GPU vs Dynamo's disaggregated + cache-aware serving.
PROFILES = [
    ("single worker (naive)",      1.0,  1.0,  "prefill+decode on one GPU, no cache reuse"),
    ("+ KV-cache-aware routing",   1.8,  0.58, "route to the worker that already has the prefix"),
    ("+ disaggregated P/D",        3.1,  0.34, "prefill and decode on separate, right-sized GPUs"),
    ("+ SLO Planner autoscale",    4.4,  0.24, "scale prefill vs decode independently to hold SLOs"),
]

_MODELS = ["nemotron-3-super:120b-a12b", "nemotron-3-nano:30b-a3b"]
_TOK = {"nemotron-3-super:120b-a12b": 20.0, "nemotron-3-nano:30b-a3b": 54.0}


def installed_models() -> list[str]:
    return _MODELS


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40.0))


# A tiny SLO simulation: as concurrency rises, the Planner adds workers to hold TTFT.
def slo_curve():
    """Yield (concurrency, ttft_ms_naive, ttft_ms_dynamo, workers)."""
    for i, conc in enumerate([8, 32, 128, 512], start=1):
        naive = 120 + conc * 4.0            # TTFT balloons without disaggregation
        dynamo = 140 + i * 12               # Planner adds workers → TTFT stays flat
        workers = 2 ** i                    # prefill/decode workers the Planner spun up
        yield conc, round(naive), round(dynamo), workers


_CANNED = ("[simulated Dynamo] Prefill and decode run on separate, right-sized GPUs; "
           "requests route to the worker that already holds the KV-cache prefix; the SLO "
           "Planner scales each pool independently to hold TTFT under load. More tokens per "
           "GPU, per dollar, and per megawatt — for agents that never stop.")


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)
