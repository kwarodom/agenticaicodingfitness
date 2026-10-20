#!/usr/bin/env python3
"""PART 3 · From tokens to goodput  [INTERMEDIATE]

Raw tokens are cheap; only USEFUL tokens create value. A reasoning agent emits
long chains of thought, so a model that's cheaper PER TOKEN can be more expensive
PER RESULT if it fails more often and has to retry.

The modern metric is cost per SUCCESSFUL task (goodput). A failed attempt isn't
free — the agent retries (burning tokens again) until it succeeds, so:

    cost / success = ($/token × tokens/attempt) × expected_attempts,
    where expected_attempts = 1 / success_rate

This demo compares a fast-but-often-wrong model against a slower-but-reliable one
and shows the fast one can cost MORE per correct answer. (Illustrative constants.)

Run:  python demos/step03_goodput.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import view  # noqa: E402

# (name, $/1M tokens, tokens per attempt, success rate) — illustrative.
# Same HARD agentic task: the small model is cheaper per token but often wrong,
# so it retries (burning tokens again); the reasoning model gets it right first try.
CANDIDATES = [
    ("fast-cheap model",   0.45, 1800, 0.35),
    ("reasoning model",    0.95, 2200, 0.92),
]


def _cost_per_success(cost_mtok: float, toks: int, rate: float) -> tuple[float, float]:
    per_attempt = cost_mtok * toks / 1_000_000        # $ for one attempt
    # expected attempts to a success = 1 / rate (retry until it works)
    per_success = per_attempt / max(rate, 1e-6)
    return per_attempt, per_success


def main() -> None:
    view.banner("PART 3", "From tokens to goodput", "INTERMEDIATE")
    view.mode_line()

    print("Tokens are cheap — but you don't want tokens, you want CORRECT ANSWERS.")
    print("So measure cost per SUCCESSFUL task, not cost per token:\n")
    print("    cost/success = ($/token × tokens/attempt) ÷ success_rate\n")

    print(f"  {'model':<20}{'$/Mtok':>8}{'tok/try':>9}{'success':>9}"
          f"{'$/attempt':>11}{'$/SUCCESS':>11}")
    print("  " + "─" * 68)
    rows = []
    for name, cost_mtok, toks, rate in CANDIDATES:
        per_attempt, per_success = _cost_per_success(cost_mtok, toks, rate)
        rows.append((name, per_success))
        print(f"  {name:<20}{cost_mtok:>7.2f}${toks:>9}{rate:>8.0%}"
              f"{per_attempt:>10.5f}${per_success:>10.5f}$")
    print()

    winner = min(rows, key=lambda r: r[1])
    loser = max(rows, key=lambda r: r[1])
    print(f"  → cheaper PER TOKEN: {CANDIDATES[0][0]} · but per SUCCESS the winner is")
    print(f"    {winner[0]} (${winner[1]:.5f} vs ${loser[1]:.5f}).\n")

    print("The 'cheap' model looks cheaper per token but fails ~65% of the time; every")
    print("failure means a retry (or a wrong answer shipped). The reasoning model spends more")
    print("tokens up front and wins on the only metric that matters: cost per RESULT.\n")

    view.generate("In two sentences, why can a model that is cheaper per token still cost "
                  "more per successful task?", max_tokens=200, title="goodput vs raw tokens")

    print("\nTakeaway: goodput reframes performance around VALUE. But you can only compute")
    print("cost/success if you can measure success — that's evaluation, next.")


if __name__ == "__main__":
    main()
