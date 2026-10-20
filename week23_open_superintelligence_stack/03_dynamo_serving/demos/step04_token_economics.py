#!/usr/bin/env python3
"""PART 4 · Token economics — cost per million tokens  [ADVANCED]

At agent scale the only metric that matters is tokens per dollar (and per megawatt).
Disaggregation + cache-aware routing + SLO autoscaling compound into a dramatically
lower cost per million tokens on the SAME hardware. This demo does the arithmetic.

Run:  python demos/step04_token_economics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

# A naive single-worker baseline of $0.90 / 1M output tokens on the DGX.
BASELINE = 0.90


def main() -> None:
    view.banner("PART 4", "Token economics — cost per million tokens", "ADVANCED")
    view.mode_line()

    print("The metric for a long-running agent: $ / 1M tokens (lower is better).")
    print(f"Baseline (naive single worker) = ${BASELINE:.2f} / 1M output tokens.\n")

    print(f"  {'serving setup':<32}{'thruput':>9}{'$/1M tok':>11}{'tok/s/GPU':>12}")
    print("  " + "─" * 66)
    for name, thru, cost_ratio, _ in sim.PROFILES:
        cost = BASELINE * cost_ratio
        tps = round(54 * thru)   # relative to a ~54 tok/s Nano baseline
        print(f"  {name:<32}{thru:>8.1f}x{cost:>10.2f}${tps:>11}")
    print()
    print("The compounding win (same DGX hardware, better serving software):")
    best = sim.PROFILES[-1]
    print(f"  • throughput:  1.0x → {best[1]:.1f}x")
    print(f"  • cost/token:  ${BASELINE:.2f} → ${BASELINE*best[2]:.2f} per 1M tokens "
          f"(~{round((1-best[2])*100)}% cheaper)")
    print("  • per megawatt: more tokens/s/GPU at the same power = more tokens per MW —")
    print("    the real currency of an AI factory.\n")

    view.generate("In two sentences, why does disaggregated serving lower the cost per "
                  "million tokens for a long-running agent?", max_tokens=300,
                  title="Dynamo at scale")

    print("\nTakeaway: Dynamo makes a sovereign agent economical to run FOREVER — the")
    print("cost/token that decides whether an always-on agent is viable. That's Week 23's")
    print("serving layer: model → NIM → flywheel → Dynamo → RL → guardrails.")


if __name__ == "__main__":
    main()
