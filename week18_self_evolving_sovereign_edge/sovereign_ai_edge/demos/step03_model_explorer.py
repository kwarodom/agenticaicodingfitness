#!/usr/bin/env python3
"""PART 4 · The Gemma 4 family — and a real local bake-off  [BEGINNER]

Gemma 4 (April 2026, Apache 2.0) spans from a sub-1.5 GB phone model to a 31B
dense model that lands top-3 on Arena. The family has two architectures, and the
difference decides your hardware:

    Dense (31B)   — all params active per token → highest quality, slower, 18 GB Q4
    MoE  (26B-A4B)— only 3.8B active per token  → ~6× faster decode,   14 GB Q4

This demo prints the variant + benchmark tables, then runs the SAME prompt
through EVERY Gemma model you actually have pulled — a real, on-device bake-off
so you can compare answer quality and tokens/sec side by side.

Run:  python demos/step03_model_explorer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import edgeview  # noqa: E402

VARIANTS = [
    ("Gemma 4 E2B",        "2.3B",  "2.3B dense",  "~1.5 GB", "128K", "Phone, RPi, IoT"),
    ("Gemma 4 E4B",        "4.5B",  "4.5B dense",  "~3 GB",   "128K", "Phone, Jetson Nano"),
    ("Gemma 4 26B-A4B MoE","26B",   "3.8B (MoE)",  "~14 GB",  "256K", "Laptop, Mac, Orin NX"),
    ("Gemma 4 31B Dense",  "31B",   "31B dense",   "~18 GB",  "256K", "DGX Spark, workstation"),
]

# Benchmark showdown (MMLU Pro / AIME 2026 / LiveCodeBench v6 / GPQA Diamond).
BENCH = [
    ("Gemma 4 31B",      "85.2%", "89.2%", "80.0%", "84.3%"),
    ("Gemma 4 26B MoE",  "82.6%", "88.3%", "77.1%", "82.3%"),
    ("Gemma 4 E4B",      "69.4%", "42.5%", "52.0%", "—"),
    ("Gemma 4 E2B",      "60.0%", "37.5%", "44.0%", "—"),
    ("Llama 3.3 70B",    "~75%",  "~40%",  "~62%",  "~60%"),
]


def main() -> None:
    edgeview.banner("PART 4", "Gemma 4 family & real local bake-off", "BEGINNER")

    print("Model variants:\n")
    print(f"  {'Variant':<22} {'Params':<6} {'Active/Token':<12} {'Q4 VRAM':<9} "
          f"{'Context':<8} {'Target'}")
    print("  " + "─" * 86)
    for name, params, active, vram, ctx, target in VARIANTS:
        print(f"  {name:<22} {params:<6} {active:<12} {vram:<9} {ctx:<8} {target}")

    print("\nBenchmark showdown (higher = better):\n")
    print(f"  {'Model':<20} {'MMLU Pro':<10} {'AIME 26':<9} {'LiveCode':<10} {'GPQA-D'}")
    print("  " + "─" * 64)
    for name, mmlu, aime, lcb, gpqa in BENCH:
        print(f"  {name:<20} {mmlu:<10} {aime:<9} {lcb:<10} {gpqa}")
    print("\n  Note: Gemma 4 26B MoE (77.1% LiveCodeBench) beats Llama 3.3 70B on")
    print("  coding while using a fraction of the memory — that's the MoE payoff.\n")

    if not edgeview.require_local():
        return

    gemmas = [m for m in config.list_local_models() if "gemma" in m.lower()
              and "cloud" not in m.lower()]  # cloud variants aren't sovereign
    if not gemmas:
        gemmas = config.list_local_models()[:2]

    prompt = ("A hotel chiller trips on 'condenser high pressure'. In ONE sentence, "
              "name the single most likely root cause.")
    print(f"Real bake-off — same prompt, {len(gemmas)} local model(s), measured live:\n")
    edgeview.sovereignty_line()

    results = []
    for m in gemmas:
        out = edgeview.generate(prompt, model=m, max_tokens=600,
                                show_reasoning=False, title=f"model: {m}")
        results.append((m, out.tok_per_s))
        print()

    print("Speed summary (this machine, right now):")
    for m, tps in sorted(results, key=lambda x: -x[1]):
        print(f"  • {m:<20} ~{tps:.1f} tok/s")
    print("\nTakeaway: you don't guess which model fits — you run them locally and")
    print("measure. Same API, swap the model name, compare quality vs speed yourself.")


if __name__ == "__main__":
    main()
