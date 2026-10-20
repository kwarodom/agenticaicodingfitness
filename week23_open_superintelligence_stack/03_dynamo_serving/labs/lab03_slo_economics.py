#!/usr/bin/env python3
"""LAB 03 · Hold an SLO under a ramp, then price your tokens.

The SLO Planner's job in miniature: declare TTFT/ITL objectives, ramp
concurrency (1 → 2 → 4 parallel streams) against your ONE worker, and watch
where the SLO breaks — that break point is exactly the signal Dynamo uses to
add prefill/decode workers. Then do the token economics: measured aggregate
tok/s + the Spark's 240 W wall power → $ per million tokens.

Run:  cd /Users/altodev/Desktop/agenticaicodingfitness && \
      .venv/bin/python week23/03_dynamo_serving/labs/lab03_slo_economics.py
Env:  SLO_TTFT_MS (default 2000) · SLO_ITL_MS (default 100) · ELEC_USD_KWH (0.15)
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SLO_TTFT = float(os.environ.get("SLO_TTFT_MS", "2000"))
SLO_ITL = float(os.environ.get("SLO_ITL_MS", "100"))
KWH_USD = float(os.environ.get("ELEC_USD_KWH", "0.15"))
POWER_W = config.DGX_SPECS["DGX Spark"]["power_w"]          # 240 W
PROMPT = "List three reasons agent traffic is bursty. Be terse."


def _stream(client, **kw):
    """Open a chat stream with hidden reasoning disabled — thinking models
    (qwen3.6, gemma4, nemotron reasoning) reason silently for 10-30 s before
    token one, which reads as a fake TTFT breach. Fall back for endpoints
    that reject the OpenAI `reasoning_effort` param."""
    try:
        return client.chat.completions.create(reasoning_effort="none", **kw)
    except Exception:  # noqa: BLE001 — param unsupported → plain call
        return client.chat.completions.create(**kw)


def one_stream(client) -> dict:
    t0, ttft, toks = time.time(), None, 0
    stream = _stream(
        client, model=config.MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=80, temperature=0.0, stream=True)
    for chunk in stream:
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        extra = getattr(d, "model_extra", None) or {}
        if (d.content or "") or getattr(d, "reasoning", None) or extra.get("reasoning"):
            toks += 1
            ttft = ttft if ttft is not None else time.time() - t0
    return {"ttft_ms": (ttft or 0.0) * 1e3, "tokens": toks,
            "elapsed": time.time() - t0}


def no_endpoint() -> None:
    print("◈ no live endpoint — real commands first:")
    print("  ollama run qwen3.6:35b-a3b-q8_0                    # local worker")
    print("  export DGX_BASE_URL=http://<spark>:11434/v1        # or your Spark")
    print("  # SLO Planner for real (on the Spark — verify module names, they drift):")
    print('  uv pip install "ai-dynamo[vllm]" && python -m dynamo.frontend --http-port 8000\n')
    print("[no endpoint — showing expected output]")
    print("  conc   p-worst TTFT   mean ITL   agg tok/s   SLO(2000/100ms)")
    print("     1         410 ms    19.8 ms        50.4   ✓ ✓")
    print("     2         740 ms    21.2 ms        94.1   ✓ ✓")
    print("     4        2350 ms    26.7 ms       151.9   ⚠ TTFT breach ✓")
    print("  → the Planner's cue: TTFT broke first → add a PREFILL worker.")
    print(f"  economics @151.9 tok/s: {POWER_W}W · ${KWH_USD}/kWh → "
          f"${POWER_W*KWH_USD/(3.6*151.9):.4f} / 1M tokens (electricity only)")


def main() -> None:
    print("▣ LAB 03 · SLO ramp + token economics — one worker, real numbers")
    print(f"  endpoint: {config.safe_base_url()}   model: {config.MODEL}   "
          f"mode: {config.MODE}")
    print(f"  SLO declared: TTFT ≤ {SLO_TTFT:.0f} ms · ITL ≤ {SLO_ITL:.0f} ms\n")
    if config.MODE != "real":
        no_endpoint()
        return
    from openai import OpenAI
    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=30.0)

    best_tps = 0.0
    print(f"  {'conc':>5}{'worst TTFT':>12}{'mean ITL':>10}{'agg tok/s':>11}   SLO")
    print("  " + "─" * 50)
    for conc in (1, 2, 4):
        t0 = time.time()
        try:
            with ThreadPoolExecutor(max_workers=conc) as pool:
                results = list(pool.map(lambda _: one_stream(client), range(conc)))
        except Exception as e:  # noqa: BLE001
            # A timeout at higher concurrency IS the lesson: the one worker is
            # saturated — the hardest possible SLO breach. Log it and move on.
            print(f"  {conc:>5}  ✗ {type(e).__name__} — worker saturated; requests "
                  f"queued past the 30s client timeout.")
            print("         That total-starvation signal is exactly what the "
                  "Planner scales on.")
            continue
        wall = time.time() - t0
        ttft = max(r["ttft_ms"] for r in results)
        itls = [(r["elapsed"] * 1e3 - r["ttft_ms"]) / max(r["tokens"] - 1, 1)
                for r in results]
        itl = sum(itls) / len(itls)
        tps = sum(r["tokens"] for r in results) / wall
        best_tps = max(best_tps, tps)
        flags = (("✓" if ttft <= SLO_TTFT else "⚠ TTFT breach") + " " +
                 ("✓" if itl <= SLO_ITL else "⚠ ITL breach"))
        print(f"  {conc:>5}{ttft:>10.0f}ms{itl:>8.1f}ms{tps:>11.1f}   {flags}")

    print("\n  reading it like the Planner: TTFT breaches first → queueing at PREFILL")
    print("  → add prefill workers; ITL creeping up → DECODE pool saturating → add")
    print("  decode workers. You just generated the Planner's input signal by hand.\n")

    if best_tps <= 0:
        print("✗ no ramp level completed — endpoint too slow for the 30s client")
        print("  timeout. Try a smaller model (ollama pull gemma4:12b) and re-run.")
        return
    cost = POWER_W * KWH_USD / (3.6 * max(best_tps, 1e-6))   # $/1M tokens
    print(f"◈ economics at your best aggregate rate ({best_tps:.1f} tok/s):")
    print(f"  {POWER_W} W × ${KWH_USD}/kWh ÷ throughput → ${cost:.4f} / 1M tokens "
          f"(electricity only, Spark wall power)")
    if not config.is_sovereign():
        print("  note: you're on a CLOUD endpoint — usage-billed; the wall-power math")
        print("  applies to the box you own, not this call.")
    print("\n✓ takeaway: SLOs decide WHEN to scale; $/M-token decides IF an always-on")
    print("  agent is viable. App 09 (09_inference_economics) goes deeper on goodput.")


if __name__ == "__main__":
    main()
