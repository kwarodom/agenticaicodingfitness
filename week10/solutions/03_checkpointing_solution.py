# %% [markdown]
# # Ex 3 SOLUTION — Checkpointing + HITL + history inspection
#
# Base exercise + demonstrates the three challenge items:
# 1. `edit` scenario
# 2. history inspection via `memory.list(config)`
# 3. restart survival (save one script run, restart Python, re-invoke)

# %%
from __future__ import annotations

import sqlite3
from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


class TicketState(TypedDict):
    ticket_body: str
    draft_response: str | None
    approved: bool | None
    sent_response: str | None


def draft(state: TicketState) -> TicketState:
    prompt = f"Draft a short 2-sentence reply to:\n\n{state['ticket_body']}"
    return {"draft_response": llm.invoke(prompt).content}


def human_approval(state: TicketState) -> TicketState:
    decision = interrupt({
        "action": "approve_reply",
        "draft": state["draft_response"],
        "question": "Approve? (approve / edit / reject)",
    })
    if isinstance(decision, dict):
        if decision.get("action") == "approve":
            return {"approved": True}
        if decision.get("action") == "edit":
            return {"draft_response": decision["text"], "approved": True}
    return {"approved": False}


def send(state: TicketState) -> TicketState:
    if not state.get("approved"):
        return {"sent_response": "[REJECTED]"}
    return {"sent_response": f"SENT: {state['draft_response']}"}


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


if __name__ == "__main__":
    thread_id = "ticket-demo-edit-001"
    config = {"configurable": {"thread_id": thread_id}}

    initial = {
        "ticket_body": "My dashboard won't load since this morning.",
        "draft_response": None,
        "approved": None,
        "sent_response": None,
    }
    state = app.invoke(initial, config=config)
    print("\n[Step 1] Paused. Auto-draft:")
    print(" ", state["draft_response"])

    # Simulate a human editing the draft before sending
    human_edit = "Thanks — we're investigating, you'll hear back within 1 hour."
    state = app.invoke(
        Command(resume={"action": "edit", "text": human_edit}),
        config=config,
    )
    print("\n[Step 2] Resumed with EDIT. Final:")
    print(" ", state["sent_response"])

    # Inspect persisted state history (challenge item 2)
    print("\n[Step 3] State history for this thread:")
    for i, c in enumerate(memory.list(config)):
        print(f"  checkpoint {i}: next={c.metadata.get('step')}")

    print("\nTip: kill this script after Step 1 and rerun with just Step 2 — state resumes from DB.")
