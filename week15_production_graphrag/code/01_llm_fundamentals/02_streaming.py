#!/usr/bin/env python3
"""Lesson 01.2 — Streaming: answers that appear as they're written.

Run:  python week15/code/01_llm_fundamentals/02_streaming.py

A normal create() call makes you wait for the whole answer. Streaming prints
each chunk as it arrives — the difference between a frozen screen and a chat
that feels alive.
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a haiku about debugging."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

print()  # newline at the end
