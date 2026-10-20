#!/usr/bin/env python3
"""PART 2 · One endpoint, many models  [BEGINNER]

The gateway's first payoff: every model on the DGX is reachable through the SAME
base_url and key — you just change the `model` name. Your apps never learn IPs,
ports, or which runtime serves what. This demo calls three aliases through the one
endpoint and shows the identical SDK + curl call.

Run:  python demos/step02_unified_endpoint.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import liteview  # noqa: E402
import litesim  # noqa: E402


def main() -> None:
    liteview.banner("PART 2", "One endpoint, many models", "BEGINNER")
    liteview.mode_line()

    print("Every model is one base_url away — only the `model` name changes:\n")
    print("    from openai import OpenAI")
    print(f'    client = OpenAI(base_url="{config.LITELLM_BASE_URL}/v1", api_key="{config.LITELLM_KEY}")')
    print('    client.chat.completions.create(model="dgx-fast",  messages=[...])')
    print('    client.chat.completions.create(model="dgx-smart", messages=[...])\n')
    print("Or curl — note the SAME url + key, different model:\n")
    print(f'    curl {config.LITELLM_BASE_URL}/v1/chat/completions \\')
    print(f'      -H "Authorization: Bearer {config.LITELLM_KEY}" \\')
    print('      -d \'{"model":"dgx-tiny","messages":[{"role":"user","content":"hi"}]}\'\n')

    print(f"Aliases exposed by this gateway: {', '.join(litesim.aliases())}\n")

    for alias in ("dgx-tiny", "dgx-fast"):
        print(f"── calling '{alias}' through the gateway:")
        liteview.generate("Name one benefit of a unified model endpoint.",
                          alias=alias, max_tokens=160, show_route=True)
        print()

    print("Why this matters on a DGX:")
    print("  • Swap a model's backend (Ollama→vLLM) with no client change — just the alias.")
    print("  • A/B two models behind one name; promote the winner by editing the config.")
    print("  • Every team codes against one URL; ops owns what's behind it.")

    print("\nTakeaway: one endpoint decouples your apps from your runtimes. Next: how the")
    print("gateway load-balances one alias across multiple deployments / Sparks.")


if __name__ == "__main__":
    main()
