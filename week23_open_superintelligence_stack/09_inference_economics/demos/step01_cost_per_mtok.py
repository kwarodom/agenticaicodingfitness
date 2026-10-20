#!/usr/bin/env python3
"""PART 1 · Cost per million tokens  [BEGINNER]

The oldest AI-performance metric, straight from the deck:

    Cost per million tokens = Infra Cost / # of Tokens

We compute it two ways — an OPEN model on YOUR DGX (you pay only amortized infra)
vs a hosted CLOUD API (usage-billed) — and land on the AI-Q claim that open-on-DGX
runs at roughly HALF the cost. In SIM the tok/s and $/Mtok are illustrative
constants; in REAL we MEASURE tok/s from a live generation and derive the cost.

Run:  python demos/step01_cost_per_mtok.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import sim  # noqa: E402
import view  # noqa: E402


def main() -> None:
    view.banner("PART 1", "Cost per million tokens", "BEGINNER")
    view.mode_line()

    print("The formula (traditional AI performance):\n")
    print("    Cost per 1M tokens  =  Infra Cost  /  # of Tokens\n")
    print("Same work, two places to buy the tokens — open on YOUR box vs a cloud API:\n")

    # DGX side: measured in REAL, illustrative constant in SIM.
    if view.is_sim():
        dgx_cost = sim.cost_per_mtok(config.MODEL)
        dgx_tps = sim.tok_s(config.MODEL)
    else:
        out = view.generate("In one sentence, define 'cost per million tokens'.",
                            max_tokens=120, title="measuring tok/s on your endpoint")
        dgx_tps = out["tok_s"] or sim.tok_s(config.MODEL)
        # illustrative amortized DGX infra: a DGX at ~$3/hr / (tok/s * 3600 s) per token,
        # scaled to 1M tokens. Keeps the shape honest without quoting a real price.
        dgx_cost = round((3.0 / 3600.0) / max(dgx_tps, 1) * 1_000_000, 2)
        print()

    cloud_cost = sim.CLOUD_MTOK
    saving = round((1 - dgx_cost / cloud_cost) * 100) if cloud_cost else 0

    print(f"  {'where':<26}{'tok/s':>8}{'$ / 1M tokens':>16}")
    print("  " + "─" * 50)
    print(f"  {'cloud frontier API':<26}{60:>8}{cloud_cost:>15.2f}$")
    print(f"  {'open model · your DGX':<26}{dgx_tps:>8.0f}{dgx_cost:>15.2f}$")
    print("  " + "─" * 50)
    print(f"  → open-on-DGX is ~{saving}% cheaper per million tokens (illustrative).\n")

    print("Why: on the DGX the tokens cost only AMORTIZED INFRA (power + the box you")
    print("already own); the cloud line is a usage-billed markup. This is the AI-Q claim —")
    print("open models on your own hardware cut cost roughly in half.\n")

    print("Takeaway: cost/Mtok is the entry-level performance metric. Next: throughput —")
    print("the same tokens/sec, divided by GPUs and by MEGAWATTS.")


if __name__ == "__main__":
    main()
