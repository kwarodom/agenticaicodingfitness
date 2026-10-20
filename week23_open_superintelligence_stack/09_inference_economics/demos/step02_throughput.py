#!/usr/bin/env python3
"""PART 2 · Throughput — per GPU and per Megawatt  [INTERMEDIATE]

The deck gives throughput two denominators:

    Throughput = (# tokens/sec) / (# of GPUs)   OR   (# tokens/sec) / (Megawatts)

Dividing by GPUs tells you how well you use silicon. Dividing by MEGAWATTS tells
you how well you use POWER — and at data-center scale power is the BINDING
constraint: you run out of megawatts long before you run out of money for chips.
Tokens-per-watt is the real currency of an AI factory (App 3 · Dynamo).

Run:  python demos/step02_throughput.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import sim  # noqa: E402
import view  # noqa: E402

# illustrative serving configs: (name, aggregate tok/s, # GPUs, kilowatts drawn)
CONFIGS = [
    ("1× DGX Spark (Nano)",       54,   1,  0.24),
    ("8× GPU node (Super)",     3200,   8,  5.60),
    ("Dynamo cluster (disagg)", 42000, 64, 44.00),
]


def main() -> None:
    view.banner("PART 2", "Throughput — per GPU and per Megawatt", "INTERMEDIATE")
    view.mode_line()

    print("The formula (two denominators):\n")
    print("    Throughput = tokens/sec ÷ # of GPUs   OR   tokens/sec ÷ Megawatts\n")

    # In REAL, anchor the smallest config to the measured single-stream tok/s.
    measured = None
    if not view.is_sim():
        out = view.generate("In one sentence, why is power the real limit at scale?",
                            max_tokens=120, title="measuring tok/s on your endpoint")
        measured = out["tok_s"] or None
        print()

    print(f"  {'serving config':<26}{'tok/s':>9}{'tok/s / GPU':>13}{'tok/s / MW':>13}")
    print("  " + "─" * 62)
    for name, tps, gpus, kw in CONFIGS:
        if measured and name.startswith("1×"):
            tps = round(measured)
        per_gpu = tps / gpus
        mw = kw / 1000.0                      # kilowatts → megawatts
        per_mw = tps / mw if mw else 0.0
        print(f"  {name:<26}{tps:>9}{per_gpu:>13.0f}{per_mw:>13,.0f}")
    print()

    print("Read it two ways:")
    print("  • per GPU  — did we tune the engine well? (higher = better silicon use)")
    print("  • per MW   — how many tokens per unit of POWER? (the factory-scale metric)\n")
    print("At scale the megawatt column is the ceiling: a site has a fixed power budget, so")
    print("MORE tokens per megawatt = more product from the same grid connection. That is why")
    print("Dynamo (App 3) optimizes tokens/s/MW, not just tokens/s/GPU.\n")

    print("Takeaway: throughput is performance per RESOURCE. Next: but raw tokens aren't the")
    print("point — only USEFUL tokens are. That's goodput.")


if __name__ == "__main__":
    main()
