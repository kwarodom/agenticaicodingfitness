# PYTHON — streaming.py
#!/usr/bin/env python3

import os
from dotenv import load_dotenv

load_dotenv()
import anthropic

client = anthropic.Anthropic()

# === Streaming — tokens appear in real-time ===
print("AI: ", end="")
with client.messages.stream(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain agentic AI in 3 sentences."}
    ]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
print()  # newline at end