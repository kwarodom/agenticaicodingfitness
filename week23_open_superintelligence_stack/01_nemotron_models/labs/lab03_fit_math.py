#!/usr/bin/env python3
"""LAB 03 · Fit math + a long-context probe — will it fit YOUR Spark?

Part A (always works, no endpoint): compute, from first principles, which
Nemotron 3 tiers fit 1 or 2 DGX Sparks at each precision — the same formula the
Week 23 runbook uses: usable = mem_gb * 0.9, weights = params_B * GB_per_B * 1.18
(the 1.18 covers KV-cache/runtime overhead). Edit the constants and re-run.

Part B (endpoint up): a miniature of the 1M-context promise — bury one anomalous
line in a haystack of log noise and see if the model retrieves it.

Run:  cd <repo root> && .venv/bin/python week23/01_nemotron_models/labs/lab03_fit_math.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# ── edit these and re-run ─────────────────────────────────────────────────────
GB_PER_B = {"FP16": 2.0, "Q8/FP8": 1.06, "Q4": 0.6, "NVFP4": 0.55}   # weights GB per B params
OVERHEAD = 1.18          # KV-cache + runtime overhead multiplier
USABLE = 0.9             # fraction of memory realistically usable
SPARK_GB = config.DGX_SPECS["DGX Spark"]["memory_gb"]                # 128
FAMILY = [("Nano  30B-A3B", 30), ("Super 120B-A12B", 120), ("Ultra 550B-A55B", 550)]


def verdict(params_b: float, prec: str, n_sparks: int) -> tuple[float, bool]:
    need = params_b * GB_PER_B[prec] * OVERHEAD
    return need, need <= SPARK_GB * n_sparks * USABLE


def part_a() -> None:
    print("◈ Part A — the fit math (weights x overhead vs usable unified memory)\n")
    print(f"  usable per Spark = {SPARK_GB} GB x {USABLE} = {SPARK_GB * USABLE:.0f} GB "
          f"(x2 Sparks = {2 * SPARK_GB * USABLE:.0f} GB)\n")
    hdr = f"  {'model':<17}" + "".join(f"{p:>12}" for p in GB_PER_B) + "   verdict"
    print(hdr + "\n  " + "─" * (len(hdr) + 6))
    for name, b in FAMILY:
        cells = []
        for prec in GB_PER_B:
            need, fits1 = verdict(b, prec, 1)
            _, fits2 = verdict(b, prec, 2)
            cells.append(f"{need:>8.0f}GB {'①' if fits1 else ('②' if fits2 else '✗')}")
        # verdict: prefer ANY precision that fits ONE Spark before reaching for two
        one = next((p for p in GB_PER_B if verdict(b, p, 1)[1]), None)
        two = next((p for p in GB_PER_B if verdict(b, p, 2)[1]), None)
        best = (f"1 Spark at {one}" if one else
                f"2 Sparks at {two}" if two else "CLOUD — does not fit 2 Sparks")
        print(f"  {name:<17}" + "".join(f"{c:>12}" for c in cells) + f"   {best}")
    print("\n  ① fits one Spark · ② needs two Sparks (TP=2 over QSFP) · ✗ no fit")
    print("  ▣ run the Ultra row yourself: 550 x 0.55 x 1.18 ≈ 357 GB > 230 GB usable")
    print("    on TWO Sparks — the honest verdict for Ultra today is the CLOUD path.")
    print("    (The app's 2-Spark chapter teaches the TP=2 recipe; the fit math is yours.)")
    print("  ▣ long context eats the margin — KV-cache grows linearly; budget 10-30 GB")
    print("    extra for 100k+-token agent runs.\n")


def part_b() -> None:
    print("◈ Part B — a 4k-char needle-in-the-haystack (the 1M-context idea, in miniature)\n")
    if config.MODE != "real":
        print("  ✗ no endpoint — to run this for real:")
        print("      ollama pull nemotron-3-nano    # or DGX_CONN=cloud + nvapi- key")
        print("  [no endpoint — showing expected output]")
        print("    · ANSWER: Line 47 — chiller-3 reported 'low refrigerant pressure'.")
        print("    ✓ found the needle at depth 47/100.")
        return
    import random
    from openai import OpenAI
    rng = random.Random(20)
    depth, total = rng.randint(20, 70), 100
    lines = [f"line {i:02d}: sensor-{rng.randint(1, 9)} nominal, {20 + rng.random() * 4:.1f}C, ok"
             for i in range(total)]
    lines[depth] = f"line {depth:02d}: chiller-3 ALARM — low refrigerant pressure"
    prompt = ("Below is a maintenance log. Exactly one line is anomalous. "
              "Name the line number and the anomaly, one sentence.\n\n" + "\n".join(lines))
    print(f"  needle buried at line {depth:02d} of {total} (~{len(prompt)} chars)")
    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY,
                    timeout=45.0, max_retries=0)   # one attempt — Part A already delivered the lesson
    r = client.chat.completions.create(model=config.MODEL, max_tokens=280, temperature=0.0,
                                       messages=[{"role": "user", "content": prompt}])
    msg = r.choices[0].message
    ans = (msg.content or "").strip()
    if "</think>" in ans:                          # some servers inline the thinking
        ans = ans.rpartition("</think>")[2].strip()
    src = ""
    if not ans:                                    # thinking models may spend the budget reasoning
        ans = str(getattr(msg, "reasoning", None)
                  or (getattr(msg, "model_extra", None) or {}).get("reasoning") or "").strip()
        src = " (surfaced in the reasoning channel — the answer budget went to thinking)"
    print(f"  · ANSWER{src}: {' '.join(ans.split())[:200]}")
    print(f"  {'✓ found the needle.' if str(depth) in ans else '✗ missed it — try a bigger model.'}")
    print("  ▣ now imagine this at 1M tokens: a week of agent history, in-window,")
    print("    no lossy summarization — that is what the Mamba+MoE hybrid buys.")


def main() -> None:
    print("━" * 64)
    print("  ▣ LAB 03 · Fit math + long-context probe — do the sizing yourself")
    print("━" * 64 + "\n")
    part_a()
    part_b()
    print("\n✓ takeaway — sizing is arithmetic, not folklore: precision x params x 1.18")
    print("  against usable memory. Pick the smallest tier that passes YOUR eval.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"\n✗ request failed ({type(e).__name__}: {e}) — Part A above is still valid.")
