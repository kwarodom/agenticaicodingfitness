#!/usr/bin/env python3
"""PART 4 · Fallbacks & reliability  [INTERMEDIATE]

A single Spark can OOM, a model can be mid-reload, a node can drop. LiteLLM keeps
the gateway up with retries, cooldowns, and FALLBACK chains: if the primary alias
fails, it transparently tries the next. This demo simulates a primary failure and
watches the request fall through the chain — your app sees one successful call.

Run:  python demos/step04_fallbacks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import liteview  # noqa: E402
import litesim  # noqa: E402

FALLBACK_YAML = """\
# add to litellm_config.yaml — reliability policy
litellm_settings:
  num_retries: 2
  request_timeout: 30
  fallbacks: [{"dgx-smart": ["dgx-fast", "dgx-tiny"]}]   # try in order on failure
  context_window_fallbacks: [{"dgx-smart": ["dgx-fast"]}] # too-long prompt → smaller-ctx model
  allowed_fails: 3
  cooldown_time: 30          # park a failing deployment for 30s
"""


def main() -> None:
    liteview.banner("PART 4", "Fallbacks & reliability", "INTERMEDIATE")
    liteview.mode_line()

    print("Reliability policy in the config:\n")
    print(FALLBACK_YAML)

    print("Simulating a request to 'dgx-smart' while spark-1 (its only deployment) is down:\n")
    chain = litesim.fallback_chain("dgx-smart")
    for i, alias in enumerate(chain):
        if i < 1:
            print(f"  ⇄ try '{alias}' … ✗ deployment unavailable (cooldown 30s) → falling back")
        else:
            print(f"  ⇄ try '{alias}' … ✓ healthy — serving the request here")
            served = alias
            break
    print()
    print(f"Your app called 'dgx-smart' and got an answer — served by '{served}'. It")
    print("never saw the failure. That's the gateway absorbing reliability for you.\n")

    liteview.generate("One sentence on why fallback chains matter for on-prem serving.",
                      alias=served, max_tokens=150)

    print("\nReliability features LiteLLM gives a DGX fleet:")
    print("  • retries + exponential backoff on transient errors")
    print("  • cooldowns park a flapping deployment so traffic avoids it")
    print("  • model + context-window fallbacks keep the gateway answering")
    print("  • health checks (/health) so you SEE which deployments are live")

    print("\nTakeaway: one Spark hiccuping shouldn't page you. Fallbacks make a fleet")
    print("of local models resilient behind a single, stable endpoint. Next: hot-swap.")


if __name__ == "__main__":
    main()
