#!/usr/bin/env python3
"""PART 2 · Disaggregated serving + KV-cache-aware routing  [INTERMEDIATE]

The two halves of inference have opposite appetites: PREFILL is compute-bound, DECODE
is memory-bandwidth-bound. Dynamo runs them on separate GPU pools and routes each
request to the worker that already holds its cached prefix. This demo shows the win.

Run:  python demos/step02_disaggregated.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

DIAGRAM = """\
   request ─► [ROUTER] ──(has prefix cached?)──► DECODE pool  (memory-bound GPUs)
                  │                                  ▲
                  └──(cold prefix)──► PREFILL pool ──┘  KV-cache moved via NIXL
                                      (compute-bound GPUs)   (QSFP 200GbE across 2 Sparks)
"""


def main() -> None:
    view.banner("PART 2", "Disaggregated serving + cache-aware routing", "INTERMEDIATE")
    view.mode_line()

    print("Split inference into two right-sized pools; route smartly between them:\n")
    print(DIAGRAM)
    print("Stacking the optimizations (relative throughput ↑, cost/M-token ↓):\n")
    print(f"  {'serving setup':<32}{'throughput':>11}{'cost/tok':>10}   what changed")
    print("  " + "─" * 88)
    for name, thru, cost, what in sim.PROFILES:
        bar = "█" * max(1, round(thru / 4.4 * 20))
        print(f"  {name:<32}{thru:>10.1f}x{cost:>9.2f}x   {what}")
        print(f"  {'':<32}{bar}")
    print()
    print("The two ideas, plainly:")
    print("  • Disaggregation — prefill GPUs and decode GPUs are different hardware needs;")
    print("    separating them keeps both fully utilized instead of bottlenecking each other.")
    print("  • Cache-aware routing — an agent's 4k-token system prompt is computed ONCE and")
    print("    reused; the router sends follow-ups to the worker that already cached it.\n")

    print("Takeaway: disaggregation + cache-aware routing ≈ 3x throughput at ~1/3 the cost/token")
    print("vs a naive single worker. Next: hold latency SLOs under real load.")


if __name__ == "__main__":
    main()
