#!/usr/bin/env python3
"""PART 2 · Hybrid Mamba-Transformer MoE + 1M context — why it's efficient  [INTERMEDIATE]

Nemotron 3's architecture is the reason it's "built for long-running agents":
a hybrid Mamba-Transformer mixture-of-experts. Mamba (state-space) layers give
near-linear scaling with sequence length; Transformer layers give precise recall;
MoE activates only a few experts per token. Net: 1M-token context + high throughput,
so an agent can hold a long history cheaply.

Run:  python demos/step02_architecture.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ntsim  # noqa: E402
import ntview  # noqa: E402


def main() -> None:
    ntview.banner("PART 2", "Hybrid Mamba-Transformer MoE + 1M context", "INTERMEDIATE")
    ntview.mode_line()

    print("The three ideas, and what each buys a long-running agent:\n")
    print("  • Mamba (SSM) layers     — attention is O(n²) in sequence length; Mamba is")
    print("                             ~O(n). That's what makes a 1M-token context")
    print("                             affordable for an agent that never forgets.")
    print("  • Transformer layers     — kept for precise, content-based recall where")
    print("                             it matters (the hybrid gets both).")
    print("  • Mixture-of-Experts     — only a few experts fire per token, so a 30B")
    print("                             model does ~3B of work/token → fast decode.\n")

    nano = ntsim.spec_for("nemotron-3-nano:30b-a3b")
    print(f"Example — {nano.name}: {int(nano.total_b)}B total, only {nano.active_b}B ACTIVE/token,")
    print(f"  {nano.ctx_k//1000}M-token context, ~{nano.tok_s_spark:.0f} tok/s on one Spark.\n")

    print("Why it matters for LONG-RUNNING + SELF-EVOLVING agents:")
    print("  • 1M context = keep the whole task history, tool outputs, and recalled")
    print("    memory in-window — fewer lossy summarizations across a long run.")
    print("  • MoE efficiency = run many specialized sub-agents on one box without")
    print("    paying dense-model decode cost for each.")
    print("  • RL-post-trained (NeMo Gym) = reasoning + tool-use are trained in, not")
    print("    bolted on — see the nemo_gym_rl app (Week 23 · app 5).")

    print("\nTakeaway: the architecture is the point — Mamba+MoE is what makes a")
    print("1M-context reasoning model cheap enough to run agents for a long time.")


if __name__ == "__main__":
    main()
