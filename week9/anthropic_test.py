# PYTHON — addingmcp.py — Using Claude with MCP tools
import anthropic, os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
client = anthropic.Anthropic()

client = anthropic.Anthropic(
api_key=os.environ['ANTHROPIC_API_KEY']
)

# Recommended models:
MODEL_FAST = 'claude-haiku-4-5-20251001' # fast, cheap
MODEL_SMART = 'claude-sonnet-4-5-20250514' # higher quality

# - claude-sonnet-4-6
# - claude-opus-4-6

resp = client.messages.create(
model=MODEL_FAST,
max_tokens=256,
messages=[{'role': 'user', 'content': 'Hello!'}],
)

print(resp.content[0].text)