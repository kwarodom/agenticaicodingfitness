#!/usr/bin/env python3
"""PART 2 · DGX hardware — what actually fits in 128 GB  [BEGINNER]

Two NVIDIA desk-side machines, very different scale:

    DGX Spark   — GB10 Grace Blackwell, 128 GB unified, ~273 GB/s, ~1 PFLOP FP4
    DGX Station — GB300 Grace Blackwell Ultra, up to 784 GB, HBM3e

The number that decides which model fits — and how fast it decodes — is MEMORY,
not TOPS. LLM decode reads every active weight once per token, so single-stream
tok/s ≈ memory-bandwidth / active-bytes-per-token. This demo prints both spec
sheets and computes what fits on a Spark from the real model registry.

Run:  python demos/step02_dgx_hardware.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import dgxsim  # noqa: E402
import dgxview  # noqa: E402


def main() -> None:
    dgxview.banner("PART 2", "DGX hardware — what fits in 128 GB", "BEGINNER")

    for name, s in config.DGX_SPECS.items():
        print(f"  {name}")
        print(f"    chip       {s['chip']}")
        print(f"    memory     {s['memory_gb']} GB · {s['memory_type']}")
        print(f"    bandwidth  {s['bandwidth_gbs']} GB/s   (the decode bottleneck)")
        print(f"    compute    ~{s['fp4_tops']} FP4 TOPS")
        print(f"    cpu / nic  {s['cpu']} · {s['nic']}")
        print(f"    fits       up to ~{s['fits_params_b']}B params · {s['note']}")
        print()

    spark = config.DGX_SPECS["DGX Spark"]
    cap = spark["memory_gb"]
    print(f"What fits on a single DGX Spark ({cap} GB unified):\n")
    print(f"  {'Model':<24}{'Arch':<6}{'Quant':<8}{'VRAM':>8}{'tok/s':>8}  fits?")
    print("  " + "─" * 62)
    for spec in dgxsim.REGISTRY.values():
        fits = "✓" if spec.vram_gb < cap * 0.92 else "✗ (too big)"
        print(f"  {spec.name:<24}{spec.arch:<6}{spec.quant:<8}"
              f"{spec.vram_gb:>6.1f}GB{spec.tok_s:>7.0f} {fits}")

    print("\nNotice the MoE win: qwen3.6 is 35B total but only ~3B ACTIVE per token,")
    print("so it decodes far faster than a 32B DENSE model of similar memory — fewer")
    print("bytes read per token. On the edge, active params drive speed.")

    print("\nTakeaway: pick by memory first. A Spark comfortably runs a 35B MoE or a")
    print("70B dense at Q4; a Station runs 670B-class models in one box.")


if __name__ == "__main__":
    main()
