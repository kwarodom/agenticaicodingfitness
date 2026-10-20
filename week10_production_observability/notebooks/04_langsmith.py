# %% [markdown]
# # Ex 4 — LangSmith tracing
#
# **Goal:** turn on tracing, inspect runs, iterate a weak prompt into a strong one.
#
# **Time:** 20 minutes.
#
# **Why this matters:** you cannot fix what you cannot see. LangSmith traces
# every node's inputs, outputs, latency, token count, and state diff. Production
# agents without observability are a liability.

# %%
from __future__ import annotations

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

# %% [markdown]
# ## Enable LangSmith tracing
#
# Get a free key at <https://smith.langchain.com>. Set `LANGSMITH_API_KEY` in
# your `.env`. The env vars below must be set **before** importing langchain
# modules that you want traced — if you already imported, restart the kernel.

# %%
# Only enable tracing when a LangSmith key is present. Without a key, the
# background uploader will 401 on every run and spam your console.
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "w10-hands-on"
    print("LangSmith tracing: ON (project='w10-hands-on')")
else:
    os.environ["LANGSMITH_TRACING"] = "false"
    print("LangSmith tracing: OFF (no LANGSMITH_API_KEY)")

llm = ChatOpenAI(
    model="openai/gpt-oss-120b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)


# %% [markdown]
# ## State

# %%
class State(TypedDict):
    ticket: str
    category: str | None


# %% [markdown]
# ## Weak prompt (on purpose)

# %%
WEAK = "Categorize this ticket: {ticket}"


# %% [markdown]
# ## Strong prompt

# %%
STRONG = (
    "You are a ticket classifier. Return exactly one of: TECHNICAL, BILLING, GENERAL. "
    "Return just the word, no punctuation, no explanation.\n\n"
    "Examples:\n"
    "  'My API is down' -> TECHNICAL\n"
    "  'Where is my invoice?' -> BILLING\n"
    "  'Do you have a demo?' -> GENERAL\n\n"
    "Ticket: {ticket}"
)


# %%
def classify_weak(state: State) -> State:
    return {"category": llm.invoke(WEAK.format(**state)).content.strip()}


def classify_strong(state: State) -> State:
    return {"category": llm.invoke(STRONG.format(**state)).content.strip()}


def build(fn):
    g = StateGraph(State)
    g.add_node("classify", fn)
    g.add_edge(START, "classify")
    g.add_edge("classify", END)
    return g.compile()


# %% [markdown]
# ## Run both and compare

# %%
if __name__ == "__main__":
    tickets = [
        "My dashboard won't load.",
        "I was charged twice.",
        "Do you have a reference in retail?",
        "API returning 500s this morning.",
    ]
    print("=== Weak prompt ===")
    weak_app = build(classify_weak)
    for t in tickets:
        r = weak_app.invoke({"ticket": t, "category": None})
        print(f"  {t[:45]:<45} -> {r['category']!r}")

    print("\n=== Strong prompt ===")
    strong_app = build(classify_strong)
    for t in tickets:
        r = strong_app.invoke({"ticket": t, "category": None})
        print(f"  {t[:45]:<45} -> {r['category']!r}")

    print("\nOpen https://smith.langchain.com -> project 'w10-hands-on' to diff both.")


# %% [markdown]
# ## CHALLENGE
#
# 1. Build a small eval dataset: use `data/fake_tickets.py` `batch(20)` to get
#    20 tickets with ground-truth categories.
# 2. Run both prompts through it.
# 3. Report accuracy for each prompt.
# 4. Bonus: push the dataset to LangSmith (`client.create_dataset(...)`) and use
#    `smith`'s built-in eval UI.
