#!/usr/bin/env python3
"""PART 3 · SLO Planner — hold latency under load  [ADVANCED]

A long-running agent's traffic is spiky. The Dynamo SLO Planner watches TTFT (time to
first token) and ITL (inter-token latency) and autoscales the prefill and decode pools
independently to hold your objectives — instead of letting latency balloon.

Run:  python demos/step03_slo_planner.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

CONFIG = """\
# Dynamo SLO Planner (sketch)
planner:
  slo: { ttft_ms: 200, itl_ms: 25 }        # objectives to hold
  pools:
    prefill: { min: 1, max: 8, gpu: dgx-spark }   # compute-bound, scale on queue depth
    decode:  { min: 1, max: 8, gpu: dgx-spark }   # memory-bound, scale on active seqs
  transfer: nixl                           # move KV-cache prefill→decode (QSFP 200GbE)
"""


def main() -> None:
    view.banner("PART 3", "SLO Planner — hold latency under load", "ADVANCED")
    view.mode_line()

    print("Declare your latency objectives; the Planner scales pools to meet them:\n")
    print(CONFIG)
    print("Simulated load ramp — TTFT with a fixed single worker vs the SLO Planner:\n")
    print(f"  {'concurrency':>12}{'naive TTFT':>13}{'Dynamo TTFT':>14}{'workers':>10}")
    print("  " + "─" * 52)
    for conc, naive, dynamo, workers in sim.slo_curve():
        flag = "  ⚠️ SLO breach" if naive > 200 else ""
        ok = "  ✅" if dynamo <= 200 else ""
        print(f"  {conc:>12}{naive:>11}ms{dynamo:>12}ms{workers:>10}{flag}{ok}")
    print()
    print("Reading the table: the naive single worker blows past the 200 ms TTFT SLO as")
    print("concurrency climbs; the Planner adds prefill/decode workers so Dynamo stays flat.\n")

    print("Why this matters for LONG-RUNNING agents: they run 24/7 with bursty demand.")
    print("Autoscaling to an SLO means you provision for the objective, not the peak —")
    print("holding latency without paying for idle GPUs at the trough.\n")

    print("Takeaway: the SLO Planner turns 'hope it's fast' into a contract. Next: what all")
    print("of this does to the token economics — cost per million tokens, per GPU, per MW.")


if __name__ == "__main__":
    main()
