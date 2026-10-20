#!/usr/bin/env python3
"""Lesson 01.3 — Multi-turn memory: how a chatbot "remembers".

Run:  python week15/code/01_llm_fundamentals/03_chatbot_memory.py

THE KEY INSIGHT: the API is stateless. The model remembers nothing between
calls. A conversation only continues because YOU keep a list and resend the
whole history every time. That `messages` list IS the chatbot's short-term
memory.

Try this:
  You: My name is Aom.
  You: What's my name?      <- it knows, because "Aom" is still in `messages`

Then flip BREAK_MEMORY = True below, rerun, and watch it forget. That broken
run is the whole lesson.
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"  # cheap + fast is fine for a chatbot

# Flip to True to sabotage memory (we stop appending the assistant's replies).
BREAK_MEMORY = False

messages = []  # <-- this list is the memory


def chat(text):
    messages.append({"role": "user", "content": text})
    reply = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=messages,  # resend EVERYTHING each turn
    ).content[0].text

    if not BREAK_MEMORY:
        messages.append({"role": "assistant", "content": reply})  # remember the reply too
    return reply


if __name__ == "__main__":
    print("Chat with Claude (type 'quit' to exit).")
    print(f"BREAK_MEMORY = {BREAK_MEMORY}\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break
        if not user_input:
            continue
        print(f"\nClaude: {chat(user_input)}\n")
