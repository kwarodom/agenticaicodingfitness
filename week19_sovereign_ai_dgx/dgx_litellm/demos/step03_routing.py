#!/usr/bin/env python3
"""PART 3 · Routing & load-balancing across Sparks  [INTERMEDIATE]

When one alias has MULTIPLE deployments (e.g. the same model on two Sparks),
LiteLLM's router spreads load across them. You pick a strategy: simple-shuffle
(round-robin), usage-based (least-busy), or latency-based (fastest p50). This demo
shows each strategy's decision for the load-balanced `dgx-fast` alias.

Run:  python demos/step03_routing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import liteview  # noqa: E402
import litesim  # noqa: E402

ROUTER_YAML = """\
# add to litellm_config.yaml — two deployments of one alias, load-balanced
model_list:
  - model_name: dgx-fast
    litellm_params: {model: ollama/qwen3.6:35b-a3b-q8_0, api_base: http://spark-0:11434}
  - model_name: dgx-fast                       # same name → a second deployment
    litellm_params: {model: ollama/qwen3.6:35b-a3b-q8_0, api_base: http://spark-1:11434}

router_settings:
  routing_strategy: usage-based-routing-v2     # or simple-shuffle | latency-based-routing
  num_retries: 2
"""


def main() -> None:
    liteview.banner("PART 3", "Routing & load-balancing across Sparks", "INTERMEDIATE")
    liteview.mode_line()

    print("Two deployments of 'dgx-fast' — one per Spark — behind one alias:\n")
    print(ROUTER_YAML)

    print("How each strategy picks a deployment for 'dgx-fast':\n")
    for strat in ("simple-shuffle", "least-busy", "latency-based"):
        _, logs = litesim.route("dgx-fast", strat)
        for ln in logs:
            print(f"  {ln}")
    print()

    print("Strategy cheat sheet:")
    print("  • simple-shuffle   — round-robin / weighted. Simplest; good default.")
    print("  • usage-based (v2) — send to the deployment with the most spare TPM/RPM.")
    print("  • latency-based    — send to the lowest measured p50. Best for UX.")
    print("  • all of them respect per-deployment tpm/rpm caps + cooldowns.\n")

    print("A live call (router picks a deployment, then we serve it):")
    liteview.generate("One line: why load-balance a model across two Sparks?",
                      alias="dgx-fast", strategy="least-busy", max_tokens=150)

    print("\nTakeaway: one alias, many Sparks, automatic balancing — horizontal scale")
    print("for your sovereign serving without touching a single client. Next: fallbacks.")


if __name__ == "__main__":
    main()
