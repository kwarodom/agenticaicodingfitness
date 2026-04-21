# %% [markdown]
# # Ex 1 — Hello StateGraph
#
# **Goal:** build your first graph with typed state, a classify node, and a router.
#
# **Time:** 25 minutes.
#
# **Learning objectives**
# - Define state with `TypedDict`
# - Write nodes as state-transforming functions
# - Wire edges and compile the graph
# - Visualize architecture with `draw_ascii()`

# %%
from __future__ import annotations

import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

# %% [markdown]
# ## Model (Qwen3 Coder 480B via OpenRouter, free tier: 20 RPM / 50 RPD per model)
#
# Free email signup at [openrouter.ai](https://openrouter.ai/keys). No credit card.
# Qwen3 Coder is a 480B MoE agentic-tool-use model that handles classification and
# supervisor routing cleanly on the free tier.
#
# Alternative free models you can swap in via the same `ChatOpenAI` client:
#
# ```python
# # model="deepseek/deepseek-r1:free",   # reasoning-heavy, slower, precise
# # model="z-ai/glm-4.6:free",           # strong tool calling
# # model="meta-llama/llama-3.3-70b-instruct:free",  # solid all-rounder
# ```
#
# To use Gemini instead (requires paid tier for classroom reliability: free is 20 RPD):
#
# ```python
# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
# ```

# %%
llm = ChatOpenAI(
    model="qwen/qwen3-coder:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)


# %% [markdown]
# ## State schema

# %%
Category = Literal["TECHNICAL", "BILLING", "GENERAL"]


class SupportState(TypedDict):
    ticket_body: str
    category: Category | None
    response: str | None


# %% [markdown]
# ## Nodes
#
# **TODO:** implement `classify` so it returns one of `TECHNICAL`, `BILLING`, `GENERAL`.
# Use the `llm.invoke(prompt).content` pattern.

# %%
def classify(state: SupportState) -> SupportState:
    """Ask the LLM to pick one of 3 categories."""
    prompt = (
        "Classify this support ticket into ONE of: TECHNICAL, BILLING, GENERAL.\n"
        "Reply with ONLY the category word — no punctuation, no explanation.\n\n"
        f"Ticket: {state['ticket_body']}"
    )
    result = llm.invoke(prompt).content.strip().upper()
    if result not in ("TECHNICAL", "BILLING", "GENERAL"):
        result = "GENERAL"
    return {"category": result}


def respond(state: SupportState) -> SupportState:
    """Simple acknowledge response. In Ex 2 we'll split by specialist."""
    template = (
        f"Thanks for reaching out. Your {state['category']} "
        f"ticket has been received. Reference: T-XXXXX."
    )
    return {"response": template}


# %% [markdown]
# ## Build the graph

# %%
graph = StateGraph(SupportState)
graph.add_node("classify", classify)
graph.add_node("respond", respond)
graph.add_edge(START, "classify")
graph.add_edge("classify", "respond")
graph.add_edge("respond", END)

app = graph.compile()

# %% [markdown]
# ## Visualize the architecture

# %%
print(app.get_graph().draw_ascii())

# %% [markdown]
# ## Try it

# %%
if __name__ == "__main__":
    samples = [
        "My API is returning 500 errors since 7am.",
        "I was charged twice last month.",
        "Can I book a demo next week?",
    ]
    for s in samples:
        result = app.invoke({"ticket_body": s, "category": None, "response": None})
        print(f"\n{s}")
        print(f"  -> {result['category']}")
        print(f"  -> {result['response']}")


# %% [markdown]
# ## CHALLENGE
#
# Add a `priority` field (`LOW | MEDIUM | HIGH`) derived from keyword match
# **before** `classify`. Print the ASCII graph again to see your architecture grow.
