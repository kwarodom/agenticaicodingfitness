#!/usr/bin/env python3
"""PART 1 · The Nemotron 3 open family — pick by task  [BEGINNER]

NVIDIA Nemotron 3 is a family of open models "built for long-running, self-evolving
agents." Three sizes + specialized variants, all open weights, hybrid Mamba-
Transformer MoE with a 1M-token context. This demo prints the family, the DGX-Spark
fit (1 vs 2 Sparks), and which model to reach for per role.

Run:  python demos/step01_family.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ntsim  # noqa: E402
import ntview  # noqa: E402


def main() -> None:
    ntview.banner("PART 1", "The Nemotron 3 open family", "BEGINNER")
    ntview.mode_line()

    print("Open Models for Multi-Agent Applications (open weights, RL-post-trained):\n")
    print(f"  {'Model':<18}{'Total':>7}{'Active':>8}{'Ctx':>7}{'VRAM(NVFP4)':>13}{'tok/s':>7}{'Sparks':>8}")
    print("  " + "─" * 74)
    for s in ntsim.FAMILY:
        print(f"  {s.name:<18}{str(int(s.total_b))+'B':>7}{str(s.active_b)+'B':>8}"
              f"{str(s.ctx_k)+'K':>7}{str(s.vram_gb_nvfp4)+'GB':>13}{s.tok_s_spark:>7.0f}{s.sparks:>8}")
    print()
    print("Roles (what to reach for):")
    for s in ntsim.FAMILY:
        print(f"  • {s.name:<18} {s.role}")
    print()
    print("The multi-agent pattern (from NVIDIA's AI-Q blueprint):")
    print("  • Nano  → intent router + specialized sub-agents (fast, cheap, many of them)")
    print("  • Super → the orchestrator / main reasoning + tool-calling agent")
    print("  • Ultra → escalate the hardest multi-step tasks (needs 2 linked Sparks)")
    print()
    print("Hardware fit:")
    print("  • 1 DGX Spark (128 GB) → Nano and Super run comfortably at NVFP4.")
    print("  • Ultra (550B) → link 2 Sparks over the QSFP 200GbE cable (tensor parallel).")

    print("\nTakeaway: one open family spans fast sub-agents to frontier reasoning —")
    print("mix sizes across a multi-agent system, all on your own hardware.")


if __name__ == "__main__":
    main()
