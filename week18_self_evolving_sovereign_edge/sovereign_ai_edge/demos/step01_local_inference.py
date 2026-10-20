#!/usr/bin/env python3
"""PART 2 · Your first sovereign inference  [BEGINNER]

Sovereign AI means the compute lives where the data is. The simplest proof is to
call a real model that is running on THIS machine and watch the tokens stream out
— while confirming that the request never crossed the network.

This is a drop-in OpenAI call. The ONLY difference from a cloud call is the
base_url: it points at localhost instead of api.openai.com. Swap that one line
and your data, your prompt, and your answer all stay inside the building.

Watch: LOCAL ✓ → PROMPT → REASON (the model thinks, on-device) → ANSWER → tok/s.

Run:  python demos/step01_local_inference.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import edgeview  # noqa: E402


def main() -> None:
    edgeview.banner("PART 2", "Your first sovereign inference", "BEGINNER")
    if not edgeview.require_local():
        return

    models = config.list_local_models()
    print("Models available on YOUR hardware right now (no download, no cloud):")
    for m in models:
        print(f"  • {m}")
    print()
    print("The code below is exactly what you'd write against OpenAI — except the")
    print("base_url is local. That one line is the whole sovereignty story:\n")
    print('    from openai import OpenAI')
    print(f'    client = OpenAI(base_url="{config.BASE_URL}", api_key="…")')
    print('    client.chat.completions.create(model=..., messages=..., stream=True)\n')

    edgeview.sovereignty_line()
    edgeview.generate(
        "In two sentences, explain why running this model locally instead of in "
        "the cloud matters for data privacy.",
        max_tokens=700,
        title="hello, sovereign world",
    )

    print("\nTakeaway: nothing here touched an external server. The prompt, the")
    print("reasoning, and the answer all stayed on this machine — that is sovereign AI.")


if __name__ == "__main__":
    main()
