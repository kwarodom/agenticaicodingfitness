#!/usr/bin/env python3
"""A faithful **LiteLLM router simulator** — learn the gateway with no GPU.

Models the parts of a LiteLLM proxy that you can't see from a single call: the
model_list (friendly alias → backend deployments), the routing strategy across
multiple deployments / Sparks, fallback chains, and per-key budgets + rate limits.
Generation itself is short + simulated; the point here is the GATEWAY behaviour.
"""
from __future__ import annotations

import time

# alias → list of backend deployments (LiteLLM load-balances across these)
MODEL_LIST = {
    "dgx-fast": [
        {"backend": "ollama/qwen3.6:35b-a3b-q8_0", "spark": "spark-0", "tpm": 12000, "p50_ms": 180},
        {"backend": "ollama/qwen3.6:35b-a3b-q8_0", "spark": "spark-1", "tpm": 12000, "p50_ms": 220},
    ],
    "dgx-smart": [
        {"backend": "vllm/llama3.3:70b-nvfp4", "spark": "spark-1", "tpm": 6000, "p50_ms": 520},
    ],
    "dgx-code": [
        {"backend": "ollama/qwen3.6:35b-a3b-q8_0", "spark": "spark-0", "tpm": 12000, "p50_ms": 180},
    ],
    "dgx-tiny": [
        {"backend": "llamacpp/phi3.5:3.8b-q4_k_m", "spark": "spark-0", "tpm": 30000, "p50_ms": 70},
    ],
}

# fallback chains: if the primary alias errors, try these in order
FALLBACKS = {
    "dgx-smart": ["dgx-fast", "dgx-tiny"],
    "dgx-fast": ["dgx-tiny"],
}

# virtual keys: per-team governance, all on-prem
VKEYS = [
    {"key": "sk-ops-•••", "team": "hvac-ops", "budget_usd": 0.0, "spent_usd": 0.0,
     "rpm": 60, "models": ["dgx-fast", "dgx-tiny"]},
    {"key": "sk-research-•••", "team": "research", "budget_usd": 0.0, "spent_usd": 0.0,
     "rpm": 20, "models": ["dgx-smart", "dgx-fast"]},
]


def aliases() -> list[str]:
    return list(MODEL_LIST)


def route(alias: str, strategy: str = "least-busy", attempt: int = 0):
    """Return (deployment, log_lines) for one routing decision."""
    deps = MODEL_LIST.get(alias, [])
    logs = []
    if not deps:
        return None, [f"router: no deployment for '{alias}'"]
    if strategy == "simple-shuffle":
        chosen = deps[attempt % len(deps)]
        why = "round-robin / weighted shuffle"
    elif strategy == "latency-based":
        chosen = min(deps, key=lambda d: d["p50_ms"])
        why = f"lowest p50 latency ({chosen['p50_ms']} ms)"
    else:  # least-busy (usage-based)
        chosen = max(deps, key=lambda d: d["tpm"])
        why = f"most spare capacity ({chosen['tpm']} tpm)"
    logs.append(f"router[{strategy}]: '{alias}' → {chosen['backend']} on "
                f"{chosen['spark']}  ({why})")
    return chosen, logs


def fallback_chain(alias: str) -> list[str]:
    return [alias] + FALLBACKS.get(alias, [])


def stream_generate(prompt: str, alias: str):
    """Yield a short simulated answer at a rate implied by the alias's p50."""
    deps = MODEL_LIST.get(alias, [{"p50_ms": 200}])
    answer = ("[simulated via LiteLLM] Routed your request to a local DGX backend; "
              "the proxy gave you one OpenAI-compatible call regardless of which "
              "runtime served it. Nothing left the box.")
    for w in answer.split(" "):
        yield w + " "
        time.sleep(0.02)
