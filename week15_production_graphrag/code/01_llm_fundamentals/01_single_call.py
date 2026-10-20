#!/usr/bin/env python3
"""Lesson 01.1 — The single call: the atom of everything.

Run:  python week15/code/01_llm_fundamentals/01_single_call.py

What to notice:
  - `messages` is a list of turns; each has a role + content.
  - The reply text lives at response.content[0].text.
  - response.usage tells you exactly what you paid (tokens in / out).
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # finds the repo-root .env no matter where you run from

import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain agentic AI in 3 sentences."}
    ],
)

print(response.content[0].text)
print(f"\nTokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
