#!/usr/bin/env python3
"""Self-improvement — GEPA: Genetic-Pareto prompt evolution  (Tutorial Part 8).

Persistent memory makes an agent stateful; closed-loop SKILL.md playbooks make
it self-improving. GEPA goes further: the agent autonomously generates,
evaluates, and evolves its OWN prompts — replacing the human prompt engineer.
The Hermes literature reports GEPA reaching higher accuracy than RL fine-tuning
with up to 35× fewer rollouts.

Three phases:
  1. Reflective Mutation — the LLM analyses its output, explains failures, and
     auto-generates N improved prompt variants.
  2. System-Aware Merge  — (multi-module) keep the best parts per module.
  3. Pareto Filtering    — score variants on accuracy / cost / tokens and keep
     only the Pareto-optimal front (no variant dominated on every axis).

Variant generation + scoring are injected as callables so the Pareto machinery
(the interesting, deterministic part) runs and is testable offline.
"""
from __future__ import annotations

from typing import Callable


def pareto_filter(scores: list[dict]) -> list[dict]:
    """Keep only Pareto-optimal solutions — not dominated on ALL axes.

    Each score dict has: accuracy (higher better), cost (lower better),
    tokens (lower better). Variant A dominates B if A is at least as good on
    every axis and strictly better on at least one.
    """
    pareto = []
    for i, s in enumerate(scores):
        dominated = False
        for j, other in enumerate(scores):
            if i == j:
                continue
            at_least_as_good = (other["accuracy"] >= s["accuracy"]
                                and other["cost"] <= s["cost"]
                                and other["tokens"] <= s["tokens"])
            strictly_better = (other["accuracy"] > s["accuracy"]
                               or other["cost"] < s["cost"]
                               or other["tokens"] < s["tokens"])
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(s)
    return pareto


def run_gepa_cycle(
    skill_name: str,
    task_description: str,
    baseline_prompt: str,
    generate_variants: Callable[[str, str, int], list[dict]],
    evaluate_variant: Callable[[str, str], dict],
    on_winner: Callable[[str, str], None] | None = None,
    n_variants: int = 5,
) -> dict:
    """Generate N prompt variants, score each, select the Pareto-best, persist it.

    Returns a report describing the chosen prompt and the Pareto front — handy
    for the visualizer to explain WHY a variant won.
    """
    # Phase 1: Reflective Mutation — the LLM proposes improved variants.
    variants = generate_variants(baseline_prompt, task_description, n_variants)
    if not variants:
        return {"winner": baseline_prompt, "reason": "no variants generated",
                "pareto_front": [], "scored": []}

    # Phases 2 & 3: evaluate each variant, then keep the Pareto-optimal front.
    scored = []
    for v in variants:
        result = evaluate_variant(v["prompt"], task_description)
        scored.append({
            "variant_id": v.get("variant_id"),
            "strategy": v.get("strategy", ""),
            "prompt": v["prompt"],
            "accuracy": result["accuracy"],   # 0–1, higher better
            "cost": result["cost_usd"],        # lower better
            "tokens": result["tokens"],        # lower better
        })

    front = pareto_filter(scored)
    if not front:
        return {"winner": baseline_prompt, "reason": "empty Pareto front",
                "pareto_front": [], "scored": scored}

    # Select the highest-accuracy variant on the Pareto front.
    best = max(front, key=lambda x: x["accuracy"])
    if on_winner:
        on_winner(skill_name, best["prompt"])
    return {"winner": best["prompt"], "strategy": best["strategy"],
            "reason": f"highest accuracy on the {len(front)}-point Pareto front",
            "pareto_front": front, "scored": scored}
