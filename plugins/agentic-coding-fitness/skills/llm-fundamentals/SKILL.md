---
name: llm-fundamentals
description: "Teach the absolute basics of talking to an LLM programmatically — single message, streaming, multi-turn conversation memory, and token counting. Use when someone is new to the Claude/Anthropic API, asks 'how do I call the model from Python?', wants to understand messages/roles/tokens, or is reviewing Week 2 of the Agentic Coding Fitness course."
when_to_use: "Beginner is starting out, asks how to call an LLM from code, doesn't understand messages/roles/streaming/tokens, or is catching up on Week 2."
---

# LLM Fundamentals — Talking to the Model (Week 2)

> **The one idea:** An LLM call is just *text in → text out*. Everything else (agents, tools, multi-agent swarms) is built on top of this single primitive. Master this first.

## What you'll understand after this skill
1. How to send one message and read the reply.
2. What "tokens" are and why you count them (cost + limits).
3. **Streaming** — printing the answer word-by-word for a better feel.
4. **Multi-turn memory** — how a chatbot "remembers" earlier messages (spoiler: *you* resend them).

---

## Setup (one time)

```bash
pip install anthropic python-dotenv
```

Create a `.env` file next to your script:

```ini
ANTHROPIC_API_KEY="sk-ant-api03-...your key..."
```

`anthropic.Anthropic()` reads that key from the environment automatically — never hard-code keys in your file.

---

## 1. The single call — the atom of everything

```python
import os
from dotenv import load_dotenv
load_dotenv()
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain agentic AI in 3 sentences."}
    ],
)
print(response.content[0].text)
print(f"\nTokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
```

**Read it line by line:**
- `messages` is a list of turns. Each turn has a `role` (`"user"` or `"assistant"`) and `content`.
- `max_tokens` caps how long the *reply* can be (1 token ≈ ¾ of a word).
- `response.content` is a **list of blocks** — text is at `response.content[0].text`. (Later, tool calls show up as other block types — see the `tool-use` skill.)
- `response.usage` tells you exactly what you paid: input tokens (your prompt) + output tokens (the reply).

> 📁 Class repo: `week02_claude_api_foundations/claudeapicall.py`

---

## 2. Streaming — answers that appear as they're written

A single `create()` call makes you wait for the whole answer. Streaming prints each chunk as it arrives — the difference between a frozen screen and a chat that feels alive.

```python
with client.messages.stream(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a haiku about debugging."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
print()
```

> 📁 Class repo: `week02_claude_api_foundations/claudestreamingapi.py`

---

## 3. Multi-turn memory — how a chatbot "remembers"

**Key insight that trips everyone up:** the API is *stateless*. The model remembers nothing between calls. A conversation only continues because **you keep a list and resend the whole history every time.** That list IS the chatbot's short-term memory.

```python
messages = []  # <-- this list is the memory

def chat(text):
    messages.append({"role": "user", "content": text})
    reply = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        messages=messages,        # resend EVERYTHING each time
    ).content[0].text
    messages.append({"role": "assistant", "content": reply})  # remember the reply too
    return reply

print(chat("My name is Aom."))
print(chat("What's my name?"))   # it knows — because "Aom" is still in `messages`
```

If you forget to append the assistant's reply back into `messages`, the model gets amnesia on the next turn. Try it and watch it break — that's the lesson.

> 📁 Class repo: `week02_claude_api_foundations/claudemulti_turn.py` (note its `system="...mentor..."` prompt — the *same text* on every turn) and the interactive loop in `week07_mcp_and_skills/agent.py`

> 💸 **Cost lever you'll want soon: prompt caching.** Notice that resending the whole history means you re-send (and re-pay for) the *same* opening tokens every turn — your `system` prompt, tool definitions, retrieved docs. Anthropic lets you mark that stable prefix as cached, and a *cache hit* costs roughly **90% less** than re-processing it. You don't need it on day one, but it's the single biggest cheap win once your conversations or contexts get long. The full cost economics (caching + model routing + context compaction) live in the `models-and-patterns` skill.

---

## Mental model to carry forward

```
[ messages list ]  →  client.messages.create()  →  response.content
       ↑__________________________________________________|
              (you append the reply to keep the memory)
```

Every agent in this course is this loop with extras bolted on: **tools** (next skill), a **goal + iteration limit** (agent loops), or **many of these running together** (multi-agent systems).

---

## 🧪 Guided lab (offer this)

Two stages: a quick **Warm-up** to prove the wiring works, then a **Skill Drill** that builds the memory loop. The drill uses a tiny **MockLLM stub** so it runs at **$0, no API key** — swap in the real client at the end.

### Warm-up (5–10 min) — *make one call and read `usage`*

Have them write the single-call example from §1 and run it against the real API (or the mock below).

**Pass / fail (binary):** the script prints a non-empty reply **and** a token line like `Tokens: 14 in, 37 out` pulled from `response.usage`. If they can't point to where the input/output token counts came from, they haven't passed — that's the whole point of the warm-up.

### Skill Drill (15–30 min) — *a multi-turn memory loop that runs at $0*

Goal: build the `messages` loop **from a blank file** and prove memory works — without spending a cent. Drop in this stub so there's no API key and no cost; it "remembers" only because the loop resends the whole list:

```python
# mock_llm.py — a fake client so the drill runs at $0, no key needed
class MockUsage:
    def __init__(self, n_in, n_out): self.input_tokens, self.output_tokens = n_in, n_out

class MockBlock:
    def __init__(self, text): self.text = text

class MockResponse:
    def __init__(self, text, messages):
        self.content = [MockBlock(text)]
        n_in = sum(len(m["content"].split()) for m in messages)
        self.usage = MockUsage(n_in, len(text.split()))

class MockMessages:
    def create(self, *, model, max_tokens, messages, system=None):
        # Fake "memory": echo back any name the history ever mentioned.
        name = next((m["content"].split("name is ")[1].split(".")[0]
                     for m in messages if "name is " in m["content"]), None)
        last = messages[-1]["content"]
        reply = (f"Your name is {name}." if "my name" in last.lower() and name
                 else f"(mock) You said: {last}")
        return MockResponse(reply, messages)

class MockLLM:
    def __init__(self): self.messages = MockMessages()

client = MockLLM()   # later: import anthropic; client = anthropic.Anthropic()
```

```python
# drill.py
from mock_llm import client

messages = []  # <-- THIS list is the memory

def chat(text):
    messages.append({"role": "user", "content": text})
    resp = client.messages.create(
        model="claude-haiku-4-5", max_tokens=1000, messages=messages,
    )
    reply = resp.content[0].text
    messages.append({"role": "assistant", "content": reply})  # remember the reply
    print(f"  [{resp.usage.input_tokens} in / {resp.usage.output_tokens} out]")
    return reply

print(chat("My name is Aom."))
print(chat("What's my name?"))   # must answer "Aom" — proof memory works
```

Run order to walk them through: (1) run it, confirm turn 2 answers "Aom"; (2) **break it on purpose** — comment out the line that appends the assistant reply, re-run, watch the input-token count and behaviour change; restore it; (3) note that the input-token count *grows every turn* — that's the cost the prompt-caching callout above is about; (4) **stretch:** delete `from mock_llm import client`, set `import anthropic; client = anthropic.Anthropic()`, and run the *same loop* against the real model — it should still know the name.

**Weighted evaluation criteria**

| # | Criterion | Weight |
|---|---|---|
| 1 | `messages` is the only state; the user turn is appended **before** the call | 1 |
| 2 | The assistant reply is appended **back** into `messages` after the call | 1 |
| 3 | Turn 2 ("what's my name?") correctly recalls "Aom" | 1 |
| 4 | They can explain *why* commenting out the append breaks memory (statelessness) | 1 |
| 5 | They read `usage` and notice input tokens grow each turn (the caching motivation) | 1 |

**Pass threshold: 4 / 5 criteria.** Criteria 2 and 4 are the heart of it — a pass that misses *both* doesn't count.

End by connecting it forward: "this `messages` loop is the skeleton of every agent — next we give it tools (`tool-use`), then a goal-driven loop (`agent-loops`)."
