#!/usr/bin/env python3
# PYTHON — multi_turn.py

import os
from dotenv import load_dotenv

load_dotenv()
import anthropic

client = anthropic.Anthropic()
messages = []

def chat(user_msg):
    messages.append({"role": "user", "content": user_msg})
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        system="You are a helpful coding mentor. Be concise.",
        messages=messages
    )
    # Extract all text blocks (handling ThinkingBlock or other block types)
    assistant_msg = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    messages.append({"role": "assistant", "content": assistant_msg})
    return assistant_msg

# Multi-turn conversation
print(chat("What is a Python decorator?"))
print(chat("Show me a simple example."))
print(chat("Now show me a decorator with arguments."))
