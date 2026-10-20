#!/usr/bin/env python3
"""NeMo Gym + NeMo RL simulator — learn verifiable-reward RL with no GPU.

Provides the same trio every Week 23 app ships (installed_models / tok_s /
stream_generate) plus RL-specific helpers the demos use to fake a GRPO training
run: verifiable-reward ENVIRONMENTS, a deterministic verifier, and a reward curve
that climbs over training rounds — all canned, all $0, all no-GPU.
"""
from __future__ import annotations

import time

# Policy models you'd post-train with NeMo RL (model → backend → ~tok/s → role).
# Small policies fit on ONE DGX Spark; bigger ones use TWO Sparks over 200GbE (TP=2).
CATALOG = [
    ("nemotron-3-nano:30b-a3b",    "TensorRT-LLM", 54, "policy — fits 1 DGX Spark"),
    ("nemotron-3-super:120b-a12b", "TensorRT-LLM", 20, "policy — 2 Sparks, TP=2"),
    ("nemotron-rag",               "TensorRT-LLM", 55, "reward-model / verifier aid"),
    ("llama-3.3-70b-instruct",     "vLLM",          6, "baseline policy"),
    ("qwen3-32b",                  "SGLang",       14, "policy — structured output"),
]
_TOK = {m: t for m, _, t, _ in CATALOG}

# The verifiable-reward environments NeMo Gym provides (task → verifier → reward).
ENVIRONMENTS = [
    ("math_verify",   "exact-match on final answer", "0.0 / 1.0", "math word problems"),
    ("code_unit",     "run hidden unit tests",       "frac tests passed", "coding tasks"),
    ("tool_call",     "correct tool + args (JSON)",  "0.0 / 1.0", "agent tool-use"),
    ("json_schema",   "validates against schema",    "0.0 / 1.0", "structured output"),
    ("retrieval_qa",  "answer contains gold span",   "0.0 / 1.0", "grounded RAG"),
]


def installed_models() -> list[str]:
    """The policies 'available' for RL on the (simulated) DGX."""
    return [m for m, *_ in CATALOG[:3]]


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40))


# ── verifiable-reward helpers (what a NeMo Gym environment does) ──────────────
def reward_fn(task: dict, answer: str) -> float:
    """A deterministic verifier: 1.0 iff the answer exactly matches the gold value.

    This is the whole point of *verifiable-reward* RL — no human, no reward model,
    just a programmatic check that returns a scalar reward.
    """
    gold = str(task.get("answer", "")).strip().lower()
    got = str(answer).strip().lower()
    return 1.0 if gold and got == gold else 0.0


def reward_curve(rounds: int = 20, start: float = 0.30, end: float = 0.80) -> list[float]:
    """A plausible mean-reward curve that climbs over GRPO training rounds.

    Smooth saturating climb from `start` to `end` with a tiny deterministic wobble,
    so the training demo looks like a real (if idealized) run.
    """
    import math
    out = []
    for i in range(rounds):
        frac = 1 - math.exp(-3.0 * i / max(1, rounds - 1))     # saturating
        wobble = 0.015 * math.sin(i * 1.7)                     # deterministic jitter
        out.append(round(start + (end - start) * frac + wobble, 3))
    return out


def rollout_group(round_idx: int, group_size: int = 8) -> list[float]:
    """Score a GROUP of rollouts for one prompt at a given training round.

    GRPO samples G completions per prompt, scores each with the verifier, and uses
    the GROUP MEAN as the baseline to compute advantages. Pass-rate rises with the
    round index, so early groups mostly fail (0.0) and later ones mostly pass (1.0).
    """
    import random
    rng = random.Random(1000 + round_idx)          # deterministic per round
    p_pass = min(0.95, 0.15 + 0.045 * round_idx)   # improves each round
    return [1.0 if rng.random() < p_pass else 0.0 for _ in range(group_size)]


def advantages(rewards: list[float]) -> list[float]:
    """Group-relative advantage = reward − group mean (GRPO's baseline)."""
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    return [round(r - mean, 3) for r in rewards]


_CANNED = ("[simulated NeMo RL policy] To book the 2pm slot I call "
           "reserve(room='Orion', start='14:00', minutes=30) — one tool call, correct "
           "args, so the verifier scores this rollout reward=1.0. After GRPO the policy "
           "picks the right tool far more often than the base model did.")


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)
