#!/usr/bin/env python3
"""Lab 03 · Will this NIM fit my Spark? — the deploy-planning math, offline.

Before you `docker run` a 50 GB NIM you should know whether it can even load.
This lab is the fit-math from the DGX Spark runbook, runnable with no GPU and
no endpoint — pure arithmetic on config.DGX_SPECS:

    weights_gb = params_B x GB-per-B(precision)
    needed_gb  = weights_gb x 1.18          (KV-cache + runtime overhead)
    budget_gb  = memory_gb x 0.90           (~90% of unified memory is usable)

Then it applies it to the Nemotron 3 family and prints the second gate every
Spark deploy must pass: is the container built for aarch64 (a -dgx-spark tag)?

Run:  cd /Users/altodev/Desktop/agenticaicodingfitness && \
      .venv/bin/python week23/02_nim_microservices/labs/lab03_spark_fit_check.py
Try your own model:  … lab03_spark_fit_check.py 70 q4    (params_B, precision)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# GB of weights per B params — runbook §1.2 (weights only; x1.18 for overhead).
PRECISIONS = {"fp16": 2.0, "q8": 1.06, "q4": 0.60, "nvfp4": 0.55}
OVERHEAD = 1.18
USABLE = 0.90

# (name, total params B, note) — the models App 01/02 teach. MoE sizes are TOTAL
# params (all experts must sit in memory), even though few are active per token.
CANDIDATES = [
    ("llama-3.1-8b-instruct (-dgx-spark verified)", 8),
    ("nemotron-3-nano  30B-A3B MoE", 30),
    ("qwen3-32b NIM (-dgx-spark verified)", 32),
    ("nemotron-3-super 120B-A12B MoE", 120),
    ("nemotron-3-ultra 550B-A55B MoE", 550),
]


def budget_gb(spec_name: str = "DGX Spark") -> float:
    return config.DGX_SPECS[spec_name]["memory_gb"] * USABLE


def needed_gb(params_b: float, precision: str) -> float:
    return params_b * PRECISIONS[precision] * OVERHEAD


def fits(params_b: float, precision: str) -> bool:
    return needed_gb(params_b, precision) <= budget_gb()


def main() -> None:
    spark = config.DGX_SPECS["DGX Spark"]
    print("━" * 64)
    print("  LAB 03 — Will this NIM fit my Spark?   [beginner]")
    print("━" * 64)
    print(f"▣ box: {spark['chip']} · {spark['memory_gb']} GB unified · "
          f"{spark['bandwidth_gbs']} GB/s\n")
    print(f"  usable budget = {spark['memory_gb']} x {USABLE:.0%} = {budget_gb():.0f} GB")
    print(f"  rule: needed = params_B x GB/B x {OVERHEAD} (KV-cache/runtime overhead)")
    print("  long agentic contexts (100k+ tok) eat another 10-30 GB — leave margin.\n")

    # ── the family table, one row per model x precision ──────────────────────
    hdr = f"  {'model':<44}" + "".join(f"{p:>9}" for p in PRECISIONS)
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for name, b in CANDIDATES:
        cells = []
        for p in PRECISIONS:
            gb = needed_gb(b, p)
            cells.append(f"{'✓' if gb <= budget_gb() else '✗'}{gb:>6.0f}G")
        print(f"  {name:<44}" + "".join(f"{c:>9}" for c in cells))
    print(f"\n  ✓ = fits in {budget_gb():.0f} GB   ✗ = does not (Ultra is CLOUD-ONLY —"
          " ~357 GB even at NVFP4)\n")

    # ── your model, if you passed one ─────────────────────────────────────────
    if len(sys.argv) >= 2:
        try:
            b = float(sys.argv[1])
            p = (sys.argv[2] if len(sys.argv) > 2 else "nvfp4").lower()
            gb = needed_gb(b, p)
            ok = gb <= budget_gb()
            print(f"◈ your model — {b:g}B at {p}: needs ~{gb:.0f} GB → "
                  f"{'✓ fits' if ok else '✗ does not fit'} "
                  f"({budget_gb():.0f} GB budget)")
            if not ok:
                two = 2 * budget_gb()
                print(f"  2 linked Sparks (256 GB, TP=2): ~{two:.0f} GB budget → "
                      f"{'✓ fits' if gb <= two else '✗ still no — use the cloud NIM'}\n")
            else:
                print()
        except (ValueError, KeyError):
            print(f"◈ usage: lab03_spark_fit_check.py <params_B> <{'/'.join(PRECISIONS)}>\n")

    # ── gate 2: memory is necessary, aarch64 is the other half ───────────────
    print("▣ gate 2 — architecture. The Spark is aarch64; only NIMs with a")
    print("  -dgx-spark (ARM64+Blackwell) build run locally. Verify BEFORE pulling:")
    print("    · check the container page on build.nvidia.com / nvcr.io for the tag")
    print("    · verified examples: llama-3.1-8b-instruct, qwen3-32b")
    print("    · nemotron-3-nano NIM for Spark is [UNCERTAIN] — check nvcr.io/nim/nvidia/")
    print("      for a -dgx-spark tag at runtime; fall back to Ollama's pull or the cloud.")
    print("  An x86-only image fails at `docker run` with an exec-format error.\n")

    print("✓ Takeaway — a NIM deploy is two gates: (1) weights x precision x 1.18")
    print("  inside ~115 GB, (2) an aarch64 build. Pass both → one docker run.")


if __name__ == "__main__":
    main()
