# %% [markdown]
# # Ex 2 — Tool-calling supervisor
#
# **Goal:** build a supervisor that routes tickets to 3 specialists using the
# **April 2026 LangChain-recommended tool-calling pattern** — simpler than the
# dedicated `langgraph-supervisor` library wrapper.
#
# **Time:** 30 minutes.
#
# **Why this pattern:** LangChain's multi-agent guidance (April 2026) recommends
# building the supervisor as a tool-calling LLM directly rather than using the
# dedicated library wrapper. You get full control over context engineering,
# specialists become tools, and you stay close to the primitives.

# %%
from __future__ import annotations

import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Qwen3 Coder 480B via OpenRouter (free tier: 20 RPM / 50 RPD per model).
# See notebook 01 for the full model-swap comment and Gemini alternative.
llm = ChatOpenAI(
    model="qwen/qwen3-coder:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)


# %% [markdown]
# ## State schema

# %%
class SupervisorState(TypedDict):
    ticket_body: str
    messages: Annotated[list, operator.add]
    final_response: str | None


# %% [markdown]
# ## Specialist tools
#
# Each specialist is a `@tool`. The supervisor picks exactly one to call.

# %%
@tool
def technical_specialist(ticket: str) -> str:
    """Handle TECHNICAL support questions (APIs, devices, dashboards, syncs, errors)."""
    return llm.invoke(
        f"You are a senior technical support engineer. Answer crisply in 2-3 sentences:\n\n{ticket}"
    ).content


@tool
def billing_specialist(ticket: str) -> str:
    """Handle BILLING questions (charges, invoices, plans, refunds, billing emails)."""
    return llm.invoke(
        f"You are a billing specialist. Be precise and reassuring in 2-3 sentences:\n\n{ticket}"
    ).content


@tool
def general_specialist(ticket: str) -> str:
    """Handle GENERAL or pre-sales questions (SLA, references, demos, integrations, certifications)."""
    return llm.invoke(
        f"You are a friendly pre-sales specialist. Be helpful and concise:\n\n{ticket}"
    ).content


SPECIALISTS = [technical_specialist, billing_specialist, general_specialist]


# %% [markdown]
# ## Supervisor agent
#
# The `create_react_agent` gives us a tool-calling ReAct loop for free.

# %%
supervisor_prompt = (
    "You are the support supervisor. Read the ticket, pick exactly ONE specialist tool, "
    "call it with the original ticket text, then stop. Do not answer the ticket yourself."
)

supervisor = create_react_agent(
    model=llm,
    tools=SPECIALISTS,
    prompt=supervisor_prompt,
)


# %% [markdown]
# ## Build graph

# %%
def run_supervisor(state: SupervisorState) -> SupervisorState:
    result = supervisor.invoke({"messages": [("user", state["ticket_body"])]})
    final = result["messages"][-1].content
    return {"final_response": final, "messages": result["messages"]}


graph = StateGraph(SupervisorState)
graph.add_node("supervisor", run_supervisor)
graph.add_edge(START, "supervisor")
graph.add_edge("supervisor", END)
app = graph.compile()

print(app.get_graph().draw_ascii())


# %% [markdown]
# ## Try it

# %%
if __name__ == "__main__":
    samples = [
        "My dashboard won't load after the update.",
        "I was charged twice for last month's plan.",
        "Do you have a reference customer in retail?",
    ]
    for s in samples:
        out = app.invoke({
            "ticket_body": s,
            "messages": [],
            "final_response": None,
        })
        print("\n---")
        print(f"Ticket:  {s}")
        print(f"Answer:  {out['final_response'][:200]}...")


# %% [markdown]
# ## CHALLENGE
#
# Add a 4th specialist `escalation_specialist` that runs when the ticket contains
# profanity or the phrase "cancel my account". Extend the supervisor prompt to
# route to it.
