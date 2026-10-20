#!/usr/bin/env python3
"""Checkpoint 6 — Self-improvement: GEPA prompt evolution  (Part 8).

GEPA (Genetic-Pareto Prompt Optimizer) lets the agent evolve its OWN prompts:
  1. Reflective Mutation — generate N improved prompt variants
  2. (System-Aware Merge for multi-module agents)
  3. Pareto Filtering    — keep only variants not dominated on accuracy / cost /
     tokens, then pick the highest-accuracy survivor

The interesting, deterministic part is the Pareto machinery — exercised here with
stub variant-generation/scoring so it runs OFFLINE. In production those callbacks
are real LLM calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from self_evolving_agent.core.gepa import pareto_filter, run_gepa_cycle


def main() -> None:
    print("● Checkpoint 6 — GEPA: Genetic-Pareto prompt evolution\n")

    # 1. Pareto filter in isolation — drop dominated variants
    scores = [
        {"id": "A", "accuracy": 0.90, "cost": 0.10, "tokens": 1000},  # baseline
        {"id": "B", "accuracy": 0.95, "cost": 0.08, "tokens": 900},   # dominates A
        {"id": "C", "accuracy": 0.93, "cost": 0.20, "tokens": 1500},  # dominated by B
        {"id": "D", "accuracy": 0.85, "cost": 0.04, "tokens": 600},   # cheap, non-dominated
    ]
    front = pareto_filter(scores)
    ids = sorted(s["id"] for s in front)
    print(f"  Pareto filter ..... 4 variants → front {ids}  (B dominates A & C; D survives on cost)")
    assert ids == ["B", "D"]

    # 2. A full GEPA cycle with stub generate/evaluate callbacks
    def generate_variants(baseline, task, n):
        # The LLM would reflect on failures; here we emit N labelled mutations.
        strategies = ["chain-of-thought", "few-shot examples", "constraint injection",
                      "role priming", "step-by-step decomposition"]
        return [{"variant_id": i, "strategy": strategies[i % len(strategies)],
                 "prompt": f"{baseline} [{strategies[i % len(strategies)]}]"}
                for i in range(n)]

    def evaluate_variant(prompt, task):
        # Deterministic scoring: CoT is most accurate, constraint injection cheapest.
        acc = 0.80 + 0.03 * ("chain-of-thought" in prompt) + 0.02 * ("step-by-step" in prompt)
        cost = 0.10 - 0.04 * ("constraint injection" in prompt)
        tokens = 1200 - 300 * ("few-shot" not in prompt)
        return {"accuracy": round(acc, 3), "cost_usd": round(cost, 3), "tokens": tokens}

    winners: list = []
    report = run_gepa_cycle(
        skill_name="analyse-codebase",
        task_description="analyse a repo and report complexity hotspots",
        baseline_prompt="Analyse the codebase.",
        generate_variants=generate_variants,
        evaluate_variant=evaluate_variant,
        on_winner=lambda name, prompt: winners.append((name, prompt)),
        n_variants=5,
    )
    print(f"\n  GEPA cycle ........ {len(report['scored'])} variants scored → "
          f"{len(report['pareto_front'])}-point Pareto front")
    print(f"  Winner strategy ... {report['strategy']}  ({report['reason']})")
    print(f"  Winning prompt .... {report['winner']}")
    assert "chain-of-thought" in report["winner"]   # highest accuracy on the front
    assert winners and winners[0][0] == "analyse-codebase"

    print("\n✓ Checkpoint 6 passed — the agent generated, scored, and Pareto-"
          "selected its own prompt, with no human prompt engineer in the loop.")


if __name__ == "__main__":
    main()
