#!/usr/bin/env python3
"""Lab 01 · Router economics — classify real queries, then do the cost math.

The demo (demos/step01_intent_router.py) routes two canned queries. This lab is
YOUR router: it classifies six queries shallow/deep against the live endpoint,
scores accuracy, and computes the blended-cost saving — the honest version of
NVIDIA's "~50% cheaper" routing claim, on your own traffic mix.

Run:  cd /Users/altodev/Desktop/agenticaicodingfitness
      .venv/bin/python week23/05_aiq_research_lab/labs/lab01_router_economics.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

QUERIES = [
    ("Who founded NVIDIA?", "shallow"),
    ("What port does Ollama listen on by default?", "shallow"),
    ("Compare the 3-year TCO of a DGX Spark vs a frontier API for a "
     "10-analyst research team.", "deep"),
    ("Survey the open-weights reasoning models released this year and rank "
     "them for on-prem legal research, with citations.", "deep"),
    ("How much unified memory does a DGX Spark have?", "shallow"),
    ("Design an evaluation to verify AI-Q's ~50% cost-reduction claim on our "
     "own traffic.", "deep"),
]
DEEP_COST_X = 9.0   # 1 deep run ≈ 9 shallow answers — the demo's Nano-vs-deep ratio
BUDGET_S = 45.0     # stop early past this — a lab must end, like a good meeting

ROUTER_PROMPT = ("You are the AI-Q Intent Router. Answer with exactly one word — "
                 "'shallow' for a simple single-source lookup, 'deep' for a multi-step, "
                 "multi-source research question that needs a plan. Query: {q}")


def _extract(text: str) -> str | None:
    clean = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    hit = None
    for m in re.finditer(r"shallow|deep", clean, re.I):
        hit = m.group(0).lower()      # last mention wins — final answers come last
    return hit


def _create(client, **kw):
    """One completion. Ask thinking models to skip the preamble — a one-word
    routing verdict needs no essay (Ollama honors `reasoning_effort: none`);
    endpoints that reject the knob (400) get a clean retry without it."""
    from openai import BadRequestError
    try:
        return client.chat.completions.create(
            extra_body={"reasoning_effort": "none"}, **kw)
    except BadRequestError:
        return client.chat.completions.create(**kw)


def _route(client, q: str, budget_left: float) -> tuple[str | None, float]:
    t0 = time.time()
    # per-call timeout shrinks to whatever budget remains — the lab MUST end
    r = _create(client.with_options(timeout=max(5.0, min(25.0, budget_left))),
                model=config.MODEL, temperature=0.0, max_tokens=300,
                messages=[{"role": "user", "content": ROUTER_PROMPT.format(q=q)}])
    msg = r.choices[0].message
    text = str(getattr(msg, "reasoning", "") or "") + " " + (msg.content or "")
    return _extract(text), time.time() - t0


def _economics(shallow_n: int, total: int) -> None:
    blended = (shallow_n * 1.0 + (total - shallow_n) * DEEP_COST_X) / total
    saved = 1.0 - blended / DEEP_COST_X
    print(f"\n  ◆ routing economics on this mix (deep run = {DEEP_COST_X:.0f}x a shallow answer):")
    print(f"    all-deep cost : {DEEP_COST_X:.1f} units/query")
    print(f"    routed cost   : {blended:.1f} units/query — {shallow_n}/{total} stayed shallow")
    print(f"    saving        : {saved:.0%} — NVIDIA's '~50%' is exactly this math on THEIR mix")


def _expected() -> None:
    print("◈ [no endpoint — showing expected output]  go REAL with any of:")
    print("    ollama pull nemotron-3-nano                     # C-path — this laptop")
    print("    export DGX_BASE_URL=http://<spark>:11434/v1     # A-path — your Spark")
    print("    export DGX_CONN=cloud DGX_API_KEY=nvapi-...     # B-path — build.nvidia.com\n")
    sample = ["shallow", "shallow", "deep", "deep", "shallow", "deep"]
    for (q, want), got in zip(QUERIES, sample):
        print(f"  ✓ model: {got.upper():7s} expected: {want.upper():7s} ~1.2s  « {q[:52]}")
    print("\n  ◆ router accuracy: 6/6 — tune the prompt (the NAT Optimizer's job) if it misses")
    _economics(3, len(QUERIES))


def main() -> None:
    print("▣ Lab 01 · Router economics — AI-Q's front door, measured")
    print(f"  endpoint: {config.safe_base_url()} · model: {config.MODEL} · mode: {config.MODE}\n")
    if config.MODE != "real":
        _expected()
        return
    from openai import OpenAI
    # max_retries=0 keeps timeouts honest — the SDK's silent retries would
    # otherwise turn one 25s stall into a 75s hang past the lab's budget.
    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY,
                    timeout=25.0, max_retries=0)
    start, hits, shallow, asked = time.time(), 0, 0, 0
    for q, want in QUERIES:
        if time.time() - start > BUDGET_S:
            print(f"  … {BUDGET_S:.0f}s budget spent — scoring the {asked} routed so far.")
            break
        try:
            got, dt = _route(client, q, BUDGET_S - (time.time() - start))
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ call failed ({type(e).__name__}) — endpoint down, or the model "
                  "outran the per-call cap; scoring what we have.")
            break
        asked += 1
        hits += got == want
        shallow += got == "shallow"
        mark = "✓" if got == want else "✗"
        print(f"  {mark} model: {str(got or '—').upper():7s} expected: {want.upper():7s} "
              f"{dt:4.1f}s  « {q[:52]}")
    if asked:
        print(f"\n  ◆ router accuracy: {hits}/{asked} — tune the prompt (the NAT Optimizer's "
              "job) if it misses")
        _economics(shallow, asked)
    print("\n✓ Takeaway — routing before reasoning is a cost lever you can MEASURE, not a")
    print("  slogan. Swap QUERIES for your real traffic and rerun; the saving is your number.")


if __name__ == "__main__":
    main()
