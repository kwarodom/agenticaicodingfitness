#!/usr/bin/env python3
"""Lesson 06.1 — Sequential pipeline (assembly line).

Run:  python week15/code/06_multi_agent/01_sequential.py

Three specialist agents, each a focused system prompt. Output of one feeds the
next: Researcher → Writer → Editor. This is CrewAI's Process.sequential, but
implemented with the raw SDK so it runs with no extra installs.

(The class repo's week9/ex1_crewai_sequential.py shows the CrewAI version.)
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"


def agent(role_system_prompt, task):
    """One agent = one focused system prompt + one task."""
    return client.messages.create(
        model=MODEL, max_tokens=1024,
        system=role_system_prompt,
        messages=[{"role": "user", "content": task}],
    ).content[0].text


TOPIC = "AI-driven building energy optimization"

if __name__ == "__main__":
    print("① Researcher...")
    research = agent(
        "You are a Senior Research Analyst. Be concise and factual.",
        f"List 4 key trends in {TOPIC}, each as one bullet with a one-line why-it-matters.",
    )
    print(research, "\n")

    print("② Writer...")
    draft = agent(
        "You are a Technical Content Writer for building managers. Make complex topics accessible.",
        f"Using this research, write a punchy 150-word intro paragraph for a blog post:\n\n{research}",
    )
    print(draft, "\n")

    print("③ Editor...")
    final = agent(
        "You are a Chief Editor. Tighten for clarity and fix any awkward phrasing. Keep it ~150 words.",
        f"Polish this draft and return only the final version:\n\n{draft}",
    )
    print("=== FINAL (after 3 agents) ===")
    print(final)
