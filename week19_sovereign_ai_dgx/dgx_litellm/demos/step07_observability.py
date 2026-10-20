#!/usr/bin/env python3
"""PART 7 · Logging & observability callbacks  [ADVANCED]

The gateway is the perfect place to observe EVERYTHING: every call from every team
to every model flows through it. LiteLLM has callbacks that emit each request to a
tracer/logger of your choice — including Phoenix on your DGX (App 3). One config
block gives you fleet-wide, on-prem observability. This demo shows the wiring and
the end-to-end picture.

Run:  python demos/step07_observability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import liteview  # noqa: E402

CALLBACK_YAML = """\
# add to litellm_config.yaml — send every gateway call to Phoenix on the DGX
litellm_settings:
  callbacks: ["arize_phoenix"]          # also: otel, langfuse, prometheus, custom
  success_callback: ["arize_phoenix"]
  failure_callback: ["arize_phoenix"]

environment_variables:
  PHOENIX_COLLECTOR_ENDPOINT: http://localhost:6006   # Phoenix on the same box
"""


def main() -> None:
    liteview.banner("PART 7", "Logging & observability callbacks", "ADVANCED")
    liteview.mode_line()

    print("One config block → every call through the gateway is traced to Phoenix:\n")
    print(CALLBACK_YAML)

    print("Why the gateway is the right observability layer:")
    print("  • EVERY request (every team, model, key) passes through it → one funnel.")
    print("  • Emits OTel/gen_ai spans → Phoenix on the DGX (App 3) — traces stay on-prem.")
    print("  • Per-key + per-model spend, latency, and error rate, fleet-wide.")
    print("  • /metrics endpoint for Prometheus → Grafana dashboards on the box.\n")

    print("The complete sovereign serving + governance + observability stack:\n")
    print("   ┌─ apps / agents (App 4) ──────────────────────────────────┐")
    print("   │            one OpenAI URL + virtual key                   │")
    print("   ├─ LiteLLM gateway (THIS app) ─────────────────────────────┤")
    print("   │   route · load-balance · fallback · keys · budgets · logs │")
    print("   ├─ backends (App 1): Ollama · vLLM · llama.cpp · TRT-LLM ───┤")
    print("   │   models, optionally fine-tuned to your domain (App 2)    │")
    print("   ├─ observability: spans → Phoenix on the DGX (App 3) ───────┤")
    print("   └─ hardware: DGX Spark(s) / Station ───────────────────────┘")
    print("   ⛔ not a request, key, trace, or token leaves the building.\n")

    liteview.generate("One sentence: why put observability at the gateway layer?",
                      alias="dgx-fast", max_tokens=150)

    print("\nTakeaway: LiteLLM is the control point of a sovereign deployment — route,")
    print("govern, AND observe every local model call from one on-prem gateway.")


if __name__ == "__main__":
    main()
