#!/usr/bin/env python3
"""PART 3 · Reasoning (RLM) — watch it think, then answer  [INTERMEDIATE]

"RLM is the next thinking": since o1/R1, models are post-trained to REASON over long
chains-of-thought before answering. Nemotron 3 is a reasoning model — it emits a
private thinking channel, then the answer. This demo runs a real (or simulated)
reasoning prompt and shows REASON → ANSWER separately.

Run:  python demos/step03_reasoning.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ntview  # noqa: E402


def main() -> None:
    ntview.banner("PART 3", "Reasoning (RLM) — think, then answer", "INTERMEDIATE")
    ntview.mode_line()

    print("The reasoning timeline (why this matters):")
    print("  2022 CoT prompt trick → 2024 o1 (RL over long CoT) → 2025 DeepSeek R1")
    print("  (reasoning from pure RL) → 2026 open reasoning models like Nemotron 3.")
    print("  Test-time compute (thinking longer) became a scaling axis of its own.\n")

    print("A reasoning model separates its PRIVATE thinking from its ANSWER — watch both:\n")
    ntview.reason(
        "A hotel has 3 chillers at COP 5.1, 4.2, and 3.8. One must go offline for "
        "maintenance. Which do you take down to lose the least efficiency, and why? "
        "Answer in two sentences.",
        max_tokens=500, show_reasoning=True, title="reasoning on your DGX")

    print("\nWhy on-DGX reasoning matters for agents:")
    print("  • The reasoning trace can contain sensitive data — keeping it on your box")
    print("    means it's never sent to a third party.")
    print("  • Long reasoning = more tokens = more cost in the cloud; on your DGX it's $0.")
    print("  • You can inspect/observe the reasoning (see the dgx_observability app, W19).")

    print("\nTakeaway: Nemotron thinks before it answers — and on your DGX, that thinking")
    print("stays yours. Next: the model calling YOUR tools (a sub-agent).")


if __name__ == "__main__":
    main()
