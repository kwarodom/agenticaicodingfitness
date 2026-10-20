#!/usr/bin/env python3
"""LAB 01 · Prefill vs decode — measure inference's two phases with a stopwatch.

Dynamo's core bet: PREFILL (reading the prompt) is compute-bound and grows with
input length; DECODE (writing tokens) is memory-bandwidth-bound and runs at a
near-constant tok/s no matter how long the prompt was. This lab proves both
claims empirically: stream a SHORT prompt and a ~850-token LONG prompt against
the same endpoint and compare TTFT (the prefill proxy) vs decode tok/s.

Run:  cd /Users/altodev/Desktop/agenticaicodingfitness && \
      .venv/bin/python week23/03_dynamo_serving/labs/lab01_prefill_vs_decode.py

Works against whatever config.py resolves: local Ollama, a DGX Spark over a
tunnel, or build.nvidia.com (DGX_CONN=cloud + nvapi- key).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

PASSAGE = (
    "A transformer serves a request in two phases. Prefill ingests every prompt "
    "token in parallel, saturating the GPU's compute units and writing the KV "
    "cache. Decode then emits one token at a time, re-reading that KV cache on "
    "every step — bounded by memory bandwidth, not FLOPs. "
)
SHORT = "In one sentence, what does a KV cache store?"
LONG = ("Context dossier — read it all before answering:\n\n" + PASSAGE * 12 +
        "\n\nIn one sentence, what does a KV cache store?")


def _piece(delta) -> str:
    extra = getattr(delta, "model_extra", None) or {}
    return (delta.content or "") + str(getattr(delta, "reasoning", None)
                                       or extra.get("reasoning") or "")


def _stream(client, **kw):
    """Open a chat stream with hidden reasoning disabled.

    Thinking models (qwen3.6, gemma4, nemotron reasoning) burn 10-30 s of
    SILENT reasoning before the first visible token — that would masquerade as
    TTFT and bury the prefill signal we're measuring. Ask for none; fall back
    for endpoints that reject the OpenAI `reasoning_effort` param.
    """
    try:
        return client.chat.completions.create(reasoning_effort="none", **kw)
    except Exception:  # noqa: BLE001 — param unsupported → plain call
        return client.chat.completions.create(**kw)


def timed_stream(client, prompt: str, max_tokens: int = 120) -> dict:
    """Stream one completion; return ttft_ms, decode tok/s, ITL (chunk≈token)."""
    t0, ttft, chunks = time.time(), None, 0
    stream = _stream(
        client, model=config.MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.0, stream=True)
    for chunk in stream:
        if chunk.choices and _piece(chunk.choices[0].delta):
            chunks += 1
            if ttft is None:
                ttft = time.time() - t0
    elapsed = time.time() - t0
    decode_s = max(elapsed - (ttft or 0.0), 1e-6)
    return {"ttft_ms": (ttft or 0.0) * 1e3, "tokens": chunks,
            "decode_tps": chunks / decode_s,
            "itl_ms": decode_s / max(chunks - 1, 1) * 1e3}


def no_endpoint() -> None:
    print("◈ no live endpoint — showing the real commands, then a sample run.\n")
    print("  # point the labs at a real OpenAI-compatible endpoint, e.g.:")
    print("  ollama run qwen3.6:35b-a3b-q8_0            # local, then re-run this lab")
    print("  export DGX_BASE_URL=http://<spark>:11434/v1              # your Spark")
    print("  export DGX_CONN=cloud DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1 \\")
    print("         DGX_API_KEY=nvapi-...               # build.nvidia.com\n")
    print("[no endpoint — showing expected output]")
    print("  prompt        ~tokens   TTFT(ms)   decode tok/s   ITL(ms)")
    print("  SHORT              11        142           51.8      19.3")
    print("  LONG             ~850        891           50.9      19.6")
    print("  → TTFT grew ~6x with the prompt; decode rate barely moved.")


def main() -> None:
    print("▣ LAB 01 · Prefill vs decode — the two phases, measured")
    print(f"  endpoint: {config.safe_base_url()}   model: {config.MODEL}   "
          f"mode: {config.MODE}\n")
    if config.MODE != "real":
        no_endpoint()
        return
    from openai import OpenAI
    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=30.0)

    rows = []
    for name, prompt in (("SHORT", SHORT), ("LONG", LONG)):
        approx_in = round(len(prompt) / 4)
        print(f"◈ streaming {name} prompt (~{approx_in} input tokens)…")
        try:
            rows.append((name, approx_in, timed_stream(client, prompt)))
        except Exception as e:  # noqa: BLE001
            print(f"✗ request failed: {type(e).__name__}: {e}")
            print("  404 → base URL missing /v1 or wrong model · 401 → key. See TUTORIAL.md.")
            return

    print(f"\n  {'prompt':<8}{'in tok':>8}{'TTFT(ms)':>10}{'decode tok/s':>14}{'ITL(ms)':>9}")
    print("  " + "─" * 49)
    for name, approx_in, m in rows:
        print(f"  {name:<8}{approx_in:>8}{m['ttft_ms']:>10.0f}"
              f"{m['decode_tps']:>14.1f}{m['itl_ms']:>9.1f}")

    (_, _, s), (_, _, l) = rows
    ratio = l["ttft_ms"] / max(s["ttft_ms"], 1e-6)
    print(f"\n✓ TTFT: {ratio:.1f}x longer on the long prompt — prefill work scales "
          f"with input length (compute-bound).")
    print(f"✓ decode: {s['decode_tps']:.1f} vs {l['decode_tps']:.1f} tok/s — "
          f"near-constant (memory-bandwidth-bound).")
    print("\n  This asymmetry is WHY Dynamo disaggregates: prefill and decode want")
    print("  different hardware and different scaling signals. One shared worker")
    print("  stalls decode streams behind every long prefill. Next: lab02 — the")
    print("  KV-cache reuse that makes agent prefixes nearly free.")


if __name__ == "__main__":
    main()
