#!/usr/bin/env python3
"""PART 5 · Model management & hot-swap  [INTERMEDIATE]

A DGX has finite VRAM, so you can't keep every model resident. LiteLLM + llama-swap
let you register MANY models behind one URL and swap them in/out of VRAM on demand
— the gateway loads the target model just-in-time and unloads the idle one. Plus
aliases and wildcards keep the model_list tidy. This demo shows the pattern.

Run:  python demos/step05_hotswap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import liteview  # noqa: E402
import litesim  # noqa: E402

SWAP_YAML = """\
# llama-swap as the backend → one port, models swapped in/out of VRAM on demand
model_list:
  - model_name: dgx-*                       # wildcard: pass any model name through
    litellm_params:
      model: openai/*
      api_base: http://localhost:8090/v1    # llama-swap proxy (does the loading)

# llama-swap's own config maps names → how to launch each model server:
#   models:
#     "qwen3.6:35b": {cmd: "vllm serve ... --port 9001", proxy: "http://localhost:9001"}
#     "llama3.3:70b": {cmd: "vllm serve ... --port 9002", proxy: "http://localhost:9002"}
#   # only ONE is resident at a time; a request for the other triggers a swap.
"""


def main() -> None:
    liteview.banner("PART 5", "Model management & hot-swap", "INTERMEDIATE")
    liteview.mode_line()

    print("The problem: 128 GB can't hold every model at once. The fix: hot-swap.\n")
    print(SWAP_YAML)

    print("What happens on a request for a non-resident model:\n")
    print("  1. client calls alias 'dgx-smart' (a 70B not currently in VRAM)")
    print("  2. llama-swap unloads the idle 35B, launches the 70B server")
    print("  3. first request waits for the load (cold start ~10-30s), then serves")
    print("  4. subsequent requests are hot until another model is needed\n")

    print(f"Aliases registered behind the one gateway: {', '.join(litesim.aliases())}")
    print("  + a `dgx-*` wildcard so new models work without editing clients.\n")

    print("Management knobs LiteLLM gives you:")
    print("  • aliases + wildcards     → tidy names, no client churn")
    print("  • /model/new at runtime   → add a model to the live proxy via API")
    print("  • llama-swap / Ollama keep_alive → control what stays warm in VRAM")
    print("  • model_info (cost, ctx)  → the gateway knows each model's limits\n")

    liteview.generate("One sentence: why hot-swap models behind one endpoint on a DGX?",
                      alias="dgx-smart", max_tokens=150)

    print("\nTakeaway: hot-swap lets a single Spark 'offer' far more models than fit in")
    print("VRAM at once — one URL, many models, loaded on demand. Next: keys & budgets.")


if __name__ == "__main__":
    main()
