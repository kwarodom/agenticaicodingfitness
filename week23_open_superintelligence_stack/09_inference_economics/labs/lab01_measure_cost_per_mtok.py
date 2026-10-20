#!/usr/bin/env python3
"""Lab 01 — measure REAL tok/s from a streamed call, then derive $/M-token.

One genuine streamed generation → time-to-first-token + decode rate → turn
watts and electricity price into dollars per million tokens for YOUR box,
and compare against an illustrative cloud list price.

Run:  cd <repo root> && .venv/bin/python week23/09_inference_economics/labs/lab01_measure_cost_per_mtok.py
Override the economics:  LAB_WATTS=240 LAB_KWH_PRICE=0.15 LAB_BOX_USD=3999
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402 — inherits DGX_CONN / DGX_BASE_URL / DGX_API_KEY

WATTS = float(os.environ.get("LAB_WATTS", "240"))          # DGX Spark under load
KWH = float(os.environ.get("LAB_KWH_PRICE", "0.15"))       # $/kWh — your tariff
BOX = float(os.environ.get("LAB_BOX_USD", "3999"))         # amortize the box
YEARS = 3.0
CLOUD_MTOK = 1.80                                          # illustrative list price
PROMPT = "In three sentences, explain why tokens are the unit of AI work."


def p(line: str = "") -> None:
    print(line, flush=True)


def economics(tok_s: float, label: str) -> None:
    elec_hr = WATTS / 1000.0 * KWH                          # $ of power per hour
    amort_hr = BOX / (YEARS * 8760.0)                       # $ of box per hour
    tokens_hr = tok_s * 3600.0
    elec_mtok = elec_hr / tokens_hr * 1e6
    full_mtok = (elec_hr + amort_hr) / tokens_hr * 1e6
    p(f"◈ Step 2 — the economics ({label})")
    p(f"  power        {WATTS:.0f} W × ${KWH:.2f}/kWh          = ${elec_hr:.4f}/hour")
    p(f"  amortization ${BOX:.0f} / {YEARS:.0f}y               = ${amort_hr:.4f}/hour")
    p(f"  throughput   {tok_s:.1f} tok/s × 3600         = {tokens_hr:,.0f} tok/hour")
    p(f"  {'':<28}{'$ / 1M tokens':>14}")
    p("  " + "─" * 44)
    p(f"  {'electricity only':<28}{elec_mtok:>13.4f}$")
    p(f"  {'+ amortized box':<28}{full_mtok:>13.4f}$")
    p(f"  {'cloud list price (illus.)':<28}{CLOUD_MTOK:>13.4f}$")
    p("  " + "─" * 44)
    ratio = CLOUD_MTOK / max(full_mtok, 1e-9)
    if ratio >= 1.0:
        p(f"  → your box is ~{ratio:.1f}× cheaper per raw token — but see lab02:")
    else:
        p(f"  → at this single-stream tok/s the box is ~{1 / ratio:.1f}× PRICIER than cloud —")
        p("    batching (more tok/s from the same watts — App 03 · Dynamo) is how owners win.")
    p("    Either way, raw tokens are not the metric; cost per SUCCESSFUL task is (lab02).")


def main() -> None:
    p("━" * 64)
    p("  ▣ LAB 01 — measure tok/s, derive $/M-token")
    p("━" * 64)
    p(f"  connection: {config.CONN} ({config.conn_human()}) · {config.safe_base_url()}")
    p(f"  model: {config.MODEL} · {config.cost_note()}\n")

    if config.MODE != "real":
        p("✗ no endpoint reachable — this lab needs a live OpenAI-compatible API.")
        p("  Start one, then re-run:")
        p("    C · local laptop : ollama serve   (then: ollama pull qwen3:4b)")
        p("    A · DGX Spark    : export DGX_BASE_URL=http://<spark>:11434/v1")
        p("    B · cloud on-ramp: export DGX_CONN=cloud \\")
        p("        DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1 DGX_API_KEY=nvapi-...")
        p("\n[no endpoint — showing expected output]")
        p("  ◈ Step 1 — streaming one real generation")
        p("  · TTFT 0.42s · ~68 tok in 1.6s → 41.8 tok/s")
        p("  ◈ Step 2 — the economics (measured)")
        p("  electricity only  0.24$ /1M tok · + amortized box 1.25$ /1M tok · cloud 1.80$")
        p("  (numbers above are a SAMPLE, not a measurement)")
        return

    import openai
    from openai import OpenAI
    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY,
                    timeout=30.0, max_retries=0)

    def start_stream():
        """Full-featured first; drop the optional params if the server 400s them."""
        base = dict(model=config.MODEL, messages=[{"role": "user", "content": PROMPT}],
                    max_tokens=300, temperature=0.3, stream=True)
        try:                       # usage in-stream + curb thinking-model traces
            return client.chat.completions.create(
                **base, stream_options={"include_usage": True},
                extra_body={"reasoning_effort": "none"})
        except openai.BadRequestError:
            return client.chat.completions.create(**base)

    p("◈ Step 1 — streaming one real generation (measuring TTFT + decode rate)")
    p(f"  » {PROMPT}")
    t0, ttft, chars, text, usage_toks = time.time(), None, 0, "", None
    try:
        for chunk in start_stream():
            u = getattr(chunk, "usage", None)
            if u and getattr(u, "completion_tokens", None):
                usage_toks = u.completion_tokens
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = delta.content or ""
            thought = str(getattr(delta, "reasoning", "") or "")   # thinking models
            if (piece or thought) and ttft is None:
                ttft = time.time() - t0
            chars += len(piece) + len(thought)
            text += piece
            print(piece, end="", flush=True)
    except Exception as e:  # noqa: BLE001
        p(f"\n✗ request failed ({type(e).__name__}: {e})")
        p("  404 → base URL likely missing :PORT and /v1 · 401 → key needed (nvapi-… for cloud)")
        return
    elapsed = time.time() - t0
    if not text and chars:
        p("  [reasoning-only reply — the model spent its whole budget thinking;")
        p("   the tokens still count (and still cost), which is itself a lesson]")
    toks = usage_toks or max(1, round(chars / 4))           # honest fallback: ~4 chars/tok
    tag = "exact (usage)" if usage_toks else "estimated (~4 chars/tok)"
    tok_s = toks / max(elapsed - (ttft or 0), 1e-6)         # decode rate, prefill excluded
    p("")
    p(f"  ✓ TTFT {ttft or 0:.2f}s · {toks} tok ({tag}) in {elapsed:.1f}s → {tok_s:.1f} tok/s decode\n")

    economics(tok_s, "measured")
    if not config.is_sovereign():
        p("\n  ⚠ you measured a CLOUD endpoint — the wall-power math above is what the same")
        p("    tok/s WOULD cost on your own box; the cloud call itself is usage-billed.")
    p("\n✓ done — next: lab02 turns raw tokens into GOODPUT (cost per successful task).")


if __name__ == "__main__":
    main()
