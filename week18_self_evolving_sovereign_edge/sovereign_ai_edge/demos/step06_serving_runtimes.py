#!/usr/bin/env python3
"""PART 7 · Inference & serving — your local AI API endpoint  [INTERMEDIATE]

A model is only useful once apps can call it. The sovereign serving stack exposes
the SAME OpenAI-compatible REST API as the cloud — so every existing app, SDK,
and agent framework works unchanged, just pointed at localhost. Three runtimes
cover every scenario:

    Ollama   — one command, auto-GPU, model library      (single user, quick start)
    vLLM     — PagedAttention, ~30% more throughput        (2+ concurrent users)
    LiteLLM  — unified proxy + llama-swap hot-swapping      (many models, one URL)

This demo makes a REAL streaming call through the OpenAI SDK against the local
endpoint and measures the two numbers that matter for serving: time-to-first-token
(latency) and decode throughput (tokens/sec).

Run:  python demos/step06_serving_runtimes.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import edgeview  # noqa: E402

RUNTIMES = [
    ("Single developer, quick start", "Ollama",              "one command, auto GPU, model library"),
    ("Production, 2+ concurrent users","vLLM",               "PagedAttention, ~30% more throughput"),
    ("Multiple models, one endpoint", "LiteLLM + llama-swap","route + hot-swap models in/out of VRAM"),
    ("Raspberry Pi / CPU-only",       "llama.cpp (server)",  "no GPU required, GGUF native"),
    ("Mobile / Raspberry Pi",         "LiteRT-LM",           "optimized for mobile/IoT"),
    ("Mac (Apple Silicon)",           "Ollama (MLX) / LiteRT","Metal GPU + MTP speculative decode"),
    ("Blackwell NVFP4 production",    "vLLM + modelopt",     "native NVFP4 kernel support"),
]


def measure_serving(model: str, prompt: str) -> tuple[float, float, int]:
    """Return (ttft_seconds, tok_per_s, tokens) for one real streaming call."""
    client = edgeview._client()
    start = time.time()
    ttft = None
    chars = 0
    stream = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=400, temperature=0.3, stream=True)
    for chunk in stream:
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        piece = (d.content or "") + (edgeview._extract_reasoning(d) or "")
        if piece:
            if ttft is None:
                ttft = time.time() - start
            chars += len(piece)
    elapsed = time.time() - start
    tokens = max(1, round(chars / 4))
    return (ttft or elapsed), tokens / elapsed, tokens


def main() -> None:
    edgeview.banner("PART 7", "Inference & serving (the local API endpoint)", "INTERMEDIATE")

    print("The headline of sovereign serving — a drop-in OpenAI client, local URL:\n")
    print("    from openai import OpenAI")
    print(f'    client = OpenAI(base_url="{config.BASE_URL}", api_key="not-needed")')
    print('    r = client.chat.completions.create(model="gemma-4", messages=[...],')
    print('                                       stream=True)\n')
    print("Or with curl:")
    print(f'    curl {config.BASE_URL}/chat/completions \\')
    print('      -H "Content-Type: application/json" \\')
    print('      -d \'{"model":"%s","messages":[{"role":"user",'
          '"content":"What is NVFP4?"}]}\'\n' % config.MODEL)

    print("When to use each runtime:\n")
    print(f"  {'Scenario':<33} {'Runtime':<22} Reason")
    print("  " + "─" * 84)
    for scen, rt, why in RUNTIMES:
        print(f"  {scen:<33} {rt:<22} {why}")

    if not edgeview.require_local():
        return

    print(f"\nNow measure THIS endpoint serving {config.MODEL} live "
          "(the two numbers that matter for serving):\n")
    edgeview.sovereignty_line()
    ttft, tps, toks = measure_serving(
        config.MODEL, "Explain in 2 sentences what an OpenAI-compatible endpoint is.")
    print(f"  ◆ time-to-first-token (latency): {ttft*1000:.0f} ms")
    print(f"  ◆ decode throughput:             ~{tps:.1f} tok/s ({toks} tokens)")
    print(f"  ◆ network hops to a cloud:        0  (served from {config.BASE_URL})")
    print("\n  Tuning notes from the field:")
    print("   • <13 tok/s on a DGX Spark? Upgrade Ollama to 0.24+ (Blackwell CUDA).")
    print("   • Bind to 127.0.0.1 or a VPN only — never expose the port publicly.")
    print("   • 2+ concurrent users → switch Ollama → vLLM for PagedAttention.")

    print("\nTakeaway: sovereign serving isn't a different API — it's the SAME API,")
    print("served from your own hardware. Apps don't know the difference; your data does.")


if __name__ == "__main__":
    main()
