#!/usr/bin/env python3
"""PART 1 · Your first sovereign inference on a DGX  [BEGINNER]

The simplest proof of sovereignty: a real model answers a prompt that never
crossed the network. The only change from a cloud call is the base_url pointing
at your DGX (or this laptop). That one line is the whole story.

If a live endpoint is reachable this runs for REAL; otherwise it SIMULATES a
DGX Spark so you can still see the mechanics. Either way: cloud cost $0.0000.

Run:  python demos/step01_dgx_hello.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import dgxview  # noqa: E402


def main() -> None:
    dgxview.banner("PART 1", "Your first sovereign inference on a DGX", "BEGINNER")
    dgxview.mode_line()
    dgxview.require_runtime()

    print("The code is exactly what you'd write against OpenAI — only base_url changes:\n")
    print("    from openai import OpenAI")
    print(f'    client = OpenAI(base_url="{config.safe_base_url()}", api_key="…")')
    print(f'    client.chat.completions.create(model="{config.MODEL}", messages=[...],')
    print("                                   stream=True)\n")

    if not dgxview.is_sim():
        print("Models present on the live endpoint right now (no download, no cloud):")
        for m in config.list_local_models() or ["(none reported)"]:
            print(f"  • {m}")
        print()

    dgxview.sovereignty_line()
    dgxview.generate(
        "In two sentences, explain why running this model on our own DGX instead "
        "of a cloud API matters for data privacy.",
        max_tokens=400,
        title="hello, sovereign DGX",
    )

    print("\nTakeaway: nothing here touched an external server. Prompt, reasoning, and")
    print("answer all stayed on your hardware — that is sovereign AI on a DGX.")


if __name__ == "__main__":
    main()
