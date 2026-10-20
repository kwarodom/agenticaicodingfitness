#!/usr/bin/env python3
# PYTHON — basic_call.py
import os
from dotenv import load_dotenv

load_dotenv()
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

# === Basic single message ===
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain agentic AI in 3 sentences."}
    ]
)
print(response.content[0].text)
print(f"\nTokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")