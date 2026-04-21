# %% [markdown]
# # Ex 3 — Checkpointing + Human-in-the-Loop
#
# **Goal:** persist state to SQLite, interrupt before the final action, and
# resume from the checkpoint after a human approves.
#
# **Time:** 25 minutes.
#
# **Why this matters:** in production, agents restart. Without checkpointing,
# you lose session state. Without `interrupt()`, you can't gate risky actions
# (sending email, charging a card, firing a missile).

# %%
from __future__ import annotations

import os
import sqlite3
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

load_dotenv()

llm = ChatOpenAI(
    model="qwen/qwen3-coder:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)


# %% [markdown]
# ## State

# %%
class TicketState(TypedDict):
    ticket_body: str
    draft_response: str | None
    approved: bool | None
    sent_response: str | None


# %% [markdown]
# ## Nodes

# %%
def draft(state: TicketState) -> TicketState:
    prompt = (
        "Draft a short, polite 2-sentence reply to this support ticket:\n\n"
        f"{state['ticket_body']}"
    )
    return {"draft_response": llm.invoke(prompt).content}


def human_approval(state: TicketState) -> TicketState:
    """Pause the graph until a human approves the draft."""
    decision = interrupt({
        "action": "approve_reply",
        "draft": state["draft_response"],
        "question": "Approve sending this reply? (approve / edit / reject)",
    })
    if isinstance(decision, dict) and decision.get("action") == "approve":
        return {"approved": True}
    if isinstance(decision, dict) and decision.get("action") == "edit":
        return {"draft_response": decision["text"], "approved": True}
    return {"approved": False}


def send(state: TicketState) -> TicketState:
    if not state.get("approved"):
        return {"sent_response": "[REJECTED — not sent]"}
    return {"sent_response": f"SENT: {state['draft_response']}"}


# %% [markdown]
# ## Wire with SQLite checkpointer
#
# `SqliteSaver` writes state to `checkpoints.db` after every node.
# The graph can be restarted and resumed from any node.

# %%
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

g = StateGraph(TicketState)
g.add_node("draft", draft)
g.add_node("human_approval", human_approval)
g.add_node("send", send)
g.add_edge(START, "draft")
g.add_edge("draft", "human_approval")
g.add_edge("human_approval", "send")
g.add_edge("send", END)

app = g.compile(checkpointer=memory)


# %% [markdown]
# ## Run with interrupt + resume

# %%
if __name__ == "__main__":
    thread_id = "ticket-demo-001"
    config = {"configurable": {"thread_id": thread_id}}

    # Step 1: run until the interrupt pauses us
    initial = {
        "ticket_body": "My dashboard won't load since this morning.",
        "draft_response": None,
        "approved": None,
        "sent_response": None,
    }
    state = app.invoke(initial, config=config)
    print("\nPaused. Draft:")
    print("  ", state.get("draft_response"))

    # Step 2: simulate a human clicking "approve"
    state = app.invoke(Command(resume={"action": "approve"}), config=config)
    print("\nResumed. Final:")
    print("  ", state.get("sent_response"))


# %% [markdown]
# ## BRIDGE — connecting Ex 2 and Ex 3
#
# The graph above is a fresh draft→human→send pipeline. Ex 2 used
# `create_react_agent` for the supervisor. A natural question: **can the
# supervisor from Ex 2 also checkpoint and interrupt?**
#
# Yes. `create_react_agent` accepts a `checkpointer` argument. Compile it once
# with `SqliteSaver` and you get the same durability + HITL affordances as a
# hand-built StateGraph. Example:
#
# ```python
# from langgraph.prebuilt import create_react_agent
# from langgraph.checkpoint.sqlite import SqliteSaver
# import sqlite3
#
# conn = sqlite3.connect("supervisor_checkpoints.db", check_same_thread=False)
# supervisor_memory = SqliteSaver(conn)
#
# supervisor = create_react_agent(
#     model=llm,
#     tools=SPECIALISTS,            # from Ex 2
#     prompt=supervisor_prompt,
#     checkpointer=supervisor_memory,
# )
#
# # Now the supervisor resumes from the last tool call after a restart,
# # and you can wrap any tool to `interrupt()` for human approval.
# ```
#
# Takeaway: Ex 2 and Ex 3 aren't separate toys. They compose.


# %% [markdown]
# ## CHALLENGE
#
# 1. Re-run with the same `thread_id` and simulate `{"action": "edit", "text": "Thanks, we're on it — fix ETA 1 hour."}` instead of `approve`.
# 2. Inspect the persisted history: `for c in memory.list(config): print(c)`.
# 3. Kill the Python process after the interrupt fires, restart, and resume — prove the graph survives restart.
# 4. Apply the BRIDGE pattern above to Ex 2's supervisor and prove it resumes mid-tool-call after a restart.
