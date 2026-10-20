#!/usr/bin/env python3
"""PART 1 · What is NVIDIA Dynamo?  [BEGINNER]

A long-running agent doesn't send one prompt — it sends millions, forever. NVIDIA
Dynamo is the distributed inference SERVING framework that keeps that economical:
disaggregated prefill/decode, KV-cache-aware routing, an SLO Planner, and NIXL for
fast cross-GPU KV transfer — across 1 or many DGX Sparks.

Run:  python demos/step01_what_is_dynamo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402

PIECES = [
    ("Disaggregated P/D", "run PREFILL (read the prompt) and DECODE (write tokens) on "
                          "separate, right-sized GPU pools — each scales on its own"),
    ("KV-cache-aware routing", "send a request to the worker that ALREADY holds its "
                               "prefix in cache — huge win for agents with long system prompts"),
    ("SLO Planner", "watch TTFT / inter-token latency and autoscale prefill vs decode "
                    "workers to hold your service-level objectives under load"),
    ("NIXL", "the low-latency transfer library that moves KV-cache between GPUs/nodes "
             "(e.g. across 2 DGX Sparks over QSFP 200GbE)"),
]


def main() -> None:
    view.banner("PART 1", "What is NVIDIA Dynamo?", "BEGINNER")
    view.mode_line()

    print("Dynamo is the RUNTIME-AT-SCALE layer of Agent = Model + Harness. Its four pieces:\n")
    for name, what in PIECES:
        print(f"  • {name}")
        print(f"      {what}")
    print()
    print("Why a LONG-RUNNING agent needs this (a single NIM isn't enough at scale):")
    print("  • Agents reuse a big system prompt every turn → cache-aware routing avoids")
    print("    re-computing the same prefix millions of times.")
    print("  • Prefill is compute-bound, decode is memory-bound → disaggregating them lets")
    print("    you buy exactly the right GPUs for each and keep both busy.")
    print("  • Traffic is spiky → the SLO Planner scales pools so latency stays flat.\n")

    print("On the hardware: 1 DGX Spark for a single pool; 2 DGX Sparks over QSFP 200GbE")
    print("(NIXL moving KV-cache between them) for disaggregated prefill/decode at scale.\n")

    print("Takeaway: Dynamo is what turns 'a model that answers' into 'a fleet that serves")
    print("forever, cheaply.' Next: how disaggregation + caching actually speed things up.")


if __name__ == "__main__":
    main()
