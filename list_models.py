#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic(
    base_url="https://api.poe.com",
    api_key=os.environ.get("POE_API_KEY")
)

try:
    # Recent SDKs support this, or we might need to check handling
    models = client.models.list()
    print("Available models:")
    for m in models.data:
        print(f"- {m.id}")
except Exception as e:
    print(f"Error listing models: {e}")

# - claude-sonnet-4-6
# - claude-opus-4-6
# - claude-opus-4-5-20251101
# - claude-haiku-4-5-20251001
# - claude-sonnet-4-5-20250929
# - claude-opus-4-1-20250805
# - claude-opus-4-20250514
# - claude-sonnet-4-20250514
# - claude-3-haiku-20240307