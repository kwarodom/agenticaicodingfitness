#!/usr/bin/env python3
"""LAB 02 · KV-cache reuse + a cache-aware router you can read in one screen.

Two halves:
  1. MEASURE — send the same ~500-token agent system prompt twice (different
     final question). If your endpoint caches prefixes (Ollama does), the warm
     call's TTFT collapses: the prefix's KV cache is reused, prefill skipped.
  2. BUILD — a 20-line prefix-hash router (pure stdlib) that does what Dynamo's
     KV-aware router does at fleet scale: send each request to the worker that
     ALREADY holds its prefix, instead of recomputing it on a random worker.

Run:  cd /Users/altodev/Desktop/agenticaicodingfitness && \
      .venv/bin/python week23/03_dynamo_serving/labs/lab02_kv_cache_router.py
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SYSTEM = ("You are the operations agent for a smart hotel. Rules: never actuate "
          "equipment without approval; cite the sensor ID for every claim; keep "
          "guest data on-premises; escalate chiller faults above severity 3; "
          "answer in at most two sentences. Site context: 2 chillers, 14 AHUs, "
          "212 rooms, BMS points prefixed 'alto/'. ") * 6   # ~500 tokens — an agent's fixed prefix
QUESTIONS = ["Which rule governs chiller faults?", "How many AHUs are on site?"]


def _stream(client, **kw):
    """Open a chat stream with hidden reasoning disabled — thinking models
    (qwen3.6, gemma4, nemotron reasoning) spend 10-30 s reasoning silently
    before token one, which would swamp the prefix-cache signal. Fall back
    for endpoints that reject the OpenAI `reasoning_effort` param."""
    try:
        return client.chat.completions.create(reasoning_effort="none", **kw)
    except Exception:  # noqa: BLE001 — param unsupported → plain call
        return client.chat.completions.create(**kw)


def _ttft(client, question: str) -> float:
    t0 = time.time()
    stream = _stream(
        client, model=config.MODEL, temperature=0.0, max_tokens=60, stream=True,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": question}])
    for chunk in stream:
        if chunk.choices:
            d = chunk.choices[0].delta
            extra = getattr(d, "model_extra", None) or {}
            if (d.content or "") or getattr(d, "reasoning", None) or extra.get("reasoning"):
                for _ in stream:      # drain the rest quickly
                    pass
                return (time.time() - t0) * 1e3
    return (time.time() - t0) * 1e3


def measure() -> None:
    print("── Part 1 · cold vs warm prefix (measured) " + "─" * 21)
    if config.MODE != "real":
        print("◈ no live endpoint — real command: ollama run qwen3.6:35b-a3b-q8_0")
        print("  (or export DGX_BASE_URL / DGX_CONN=cloud — see TUTORIAL.md §0)\n")
        print("[no endpoint — showing expected output]")
        print("  call 1 (cold prefix)  TTFT ≈ 780 ms   — full prefill of ~500 tokens")
        print("  call 2 (warm prefix)  TTFT ≈ 160 ms   — prefix KV cache reused: 4.9x faster\n")
        return
    from openai import OpenAI
    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=30.0)
    print(f"  shared system prompt ≈ {round(len(SYSTEM)/4)} tokens · model {config.MODEL}")
    try:
        cold = _ttft(client, QUESTIONS[0])
        warm = _ttft(client, QUESTIONS[1])
    except Exception as e:  # noqa: BLE001
        print(f"✗ request failed: {type(e).__name__}: {e} — see TUTORIAL.md troubleshooting.")
        return
    print(f"  call 1 (cold-ish prefix)  TTFT {cold:>6.0f} ms")
    print(f"  call 2 (warm prefix)      TTFT {warm:>6.0f} ms   → {cold/max(warm,1e-6):.1f}x")
    print("  honest note: 'cold' is best-effort — if you just ran this lab, call 1")
    print("  may already be warm and the gap small. A cloud gateway may not cache.\n")


def route_demo() -> None:
    print("── Part 2 · a cache-aware router in 20 lines (stdlib) " + "─" * 10)
    workers = ["spark-0", "spark-1", "spark-2"]
    cache: dict[str, str] = {}          # prefix-hash → worker that holds its KV
    rr = 0
    print(f"  {'req':<5}{'agent prefix':<14}{'decision':<10}worker")
    print("  " + "─" * 44)
    for i, agent in enumerate(["A", "B", "A", "A", "B", "C"], 1):
        h = hashlib.sha256(f"system-prompt-of-agent-{agent}".encode()).hexdigest()[:8]
        if h in cache:
            worker, hit = cache[h], "✓ HIT"
        else:
            worker, rr = workers[rr % len(workers)], rr + 1   # miss → round-robin
            cache[h], hit = worker, "◈ MISS"
        print(f"  #{i:<4}{agent + ' (' + h + ')':<14}{hit:<10}{worker}")
    hits = 6 - len(cache)
    print(f"\n  {hits}/6 requests skipped prefill entirely ({hits/6:.0%} warm) — a naive")
    print("  round-robin router would have recomputed those prefixes on cold workers.")
    print("  Dynamo's KV-aware router tracks real cache block residency per worker;")
    print("  the routing decision above is the same shape.")


def main() -> None:
    print("▣ LAB 02 · KV-cache-aware routing — measure it, then build it")
    print(f"  endpoint: {config.safe_base_url()}   mode: {config.MODE}\n")
    measure()
    route_demo()
    print("\n✓ takeaway: an agent fleet re-sends the same long prefix millions of")
    print("  times — cache reuse (part 1) times routing (part 2) is Dynamo's ~1.8x.")
    print("  Next: lab03 — hold a latency SLO and price your tokens.")


if __name__ == "__main__":
    main()
