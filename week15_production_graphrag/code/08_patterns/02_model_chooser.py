#!/usr/bin/env python3
"""Lesson 08.2 — A tiny model chooser (the decision layer, Week 11).

Run:  python week15/code/08_patterns/02_model_chooser.py

There's no "best" model — only the best fit on four axes: capability, cost,
latency, context. This is NOT an API demo; it's a thinking tool. It encodes the
class's rules of thumb so you can pressure-test your own choices.

Edit the SCENARIOS list with your own use cases.
"""
from dataclasses import dataclass


@dataclass
class Scenario:
    name: str
    needs_hard_reasoning: bool   # complex planning / tricky code / agent "brain"?
    high_volume: bool            # many calls / many agents / tight budget?
    latency_sensitive: bool      # real-time, user-facing, big swarm?
    huge_context: bool           # long docs / big codebase / long history?


def recommend(s: Scenario) -> str:
    # Frontier tier = the brain; small/fast tier = volume & parallel work.
    if s.needs_hard_reasoning and not s.high_volume:
        base = "Frontier (e.g. Opus/Sonnet tier) — you need the reasoning."
    elif s.high_volume or s.latency_sensitive:
        base = "Small/fast (e.g. Haiku tier) — cheap & quick for volume/parallel work."
    else:
        base = "Mid (Sonnet tier) — solid default balance."

    notes = []
    if s.needs_hard_reasoning and s.high_volume:
        notes.append("MIX models: a frontier model PLANS, cheap models EXECUTE.")
    if s.huge_context:
        notes.append("Prioritize a large context window; consider RAG to avoid paying for it every call.")
    if s.latency_sensitive and s.needs_hard_reasoning:
        notes.append("Tension: reasoning vs. speed — try a fast model + the Reflection pattern.")
    return base + ("\n     ↳ " + "\n     ↳ ".join(notes) if notes else "")


SCENARIOS = [
    Scenario("Support-ticket classifier (router node)", False, True, True, False),
    Scenario("Autonomous code-fixing agent", True, False, False, True),
    Scenario("5-agent parallel building audit", False, True, True, False),
    Scenario("Answer Qs over a 500-page manual", False, False, False, True),
    Scenario("Plan a complex task, then run 100s of sub-steps", True, True, False, False),
]

if __name__ == "__main__":
    print("Model selection — rules of thumb applied to each scenario:\n")
    for s in SCENARIOS:
        print(f"• {s.name}")
        print(f"     → {recommend(s)}\n")
    print("Decision flow for a NEW project:")
    print("  1 call enough? → just call the API   (don't build an agent)")
    print("  needs to act?  → add tools           (folder 02)")
    print("  multi-step?    → agent loop          (folder 03)")
    print("  use your docs? → RAG                 (folder 05)")
    print("  distinct subtasks/parallel? → multi-agent (folder 06)")
    print("  remember across runs?       → graph memory (folder 07)")
    print("  Bias toward the SIMPLEST rung that solves the problem.")
