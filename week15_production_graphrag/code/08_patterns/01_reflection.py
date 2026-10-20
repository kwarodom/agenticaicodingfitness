#!/usr/bin/env python3
"""Lesson 08.1 — The Reflection pattern.

Run:  python week15/code/08_patterns/01_reflection.py

Reflection = the agent critiques and revises its OWN output. Draft → critique →
improve. It's one of the highest-leverage patterns: a cheap way to raise
quality without more tools or more agents.

We do one round here; loop it N times for harder tasks.
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"
TASK = "Write a 2-sentence product tagline for an AI that optimizes building energy use."


def ask(system, user):
    return client.messages.create(
        model=MODEL, max_tokens=400, system=system,
        messages=[{"role": "user", "content": user}],
    ).content[0].text


if __name__ == "__main__":
    print("① DRAFT")
    draft = ask("You are a copywriter.", TASK)
    print(draft, "\n")

    print("② CRITIQUE (the agent grades its own work)")
    critique = ask(
        "You are a ruthless creative director. Critique briefly and specifically.",
        f"Task: {TASK}\n\nDraft:\n{draft}\n\nList 3 concrete weaknesses and how to fix each.",
    )
    print(critique, "\n")

    print("③ IMPROVE (revise using the critique)")
    final = ask(
        "You are a copywriter. Apply the critique and return only the improved tagline.",
        f"Task: {TASK}\n\nYour draft:\n{draft}\n\nCritique to address:\n{critique}",
    )
    print(final)

    print("\nThat draft → critique → improve loop is the Reflection pattern. "
          "Run it 2–3× for harder tasks.")
