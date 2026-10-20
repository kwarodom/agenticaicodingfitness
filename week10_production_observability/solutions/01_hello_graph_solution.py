# %% [markdown]
# # Ex 1 SOLUTION — Hello StateGraph + priority challenge
#
# Base exercise + the `priority` field (LOW / MEDIUM / HIGH) derived from
# keyword match, inserted **before** classify.

# %%
from __future__ import annotations

from typing import Literal, TypedDict

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

llm = ChatOpenAI(model="openai/gpt-oss-120b:free", base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"), temperature=0)

Category = Literal["TECHNICAL", "BILLING", "GENERAL"]
Priority = Literal["LOW", "MEDIUM", "HIGH"]


class SupportState(TypedDict):
    ticket_body: str
    priority: Priority | None
    category: Category | None
    response: str | None


URGENT_KEYWORDS = {"down", "outage", "error", "crash", "urgent", "immediately", "500"}
MEDIUM_KEYWORDS = {"slow", "stuck", "fails", "broken", "delayed"}


def prioritize(state: SupportState) -> SupportState:
    body = state["ticket_body"].lower()
    if any(k in body for k in URGENT_KEYWORDS):
        return {"priority": "HIGH"}
    if any(k in body for k in MEDIUM_KEYWORDS):
        return {"priority": "MEDIUM"}
    return {"priority": "LOW"}


def classify(state: SupportState) -> SupportState:
    prompt = (
        "Classify this support ticket into ONE of: TECHNICAL, BILLING, GENERAL.\n"
        "Reply with ONLY the category word.\n\n"
        f"Ticket: {state['ticket_body']}"
    )
    result = llm.invoke(prompt).content.strip().upper()
    if result not in ("TECHNICAL", "BILLING", "GENERAL"):
        result = "GENERAL"
    return {"category": result}


def respond(state: SupportState) -> SupportState:
    template = (
        f"[{state['priority']}] Thanks for reaching out. "
        f"Your {state['category']} ticket has been received. Reference: T-XXXXX."
    )
    return {"response": template}


graph = StateGraph(SupportState)
graph.add_node("prioritize", prioritize)
graph.add_node("classify", classify)
graph.add_node("respond", respond)
graph.add_edge(START, "prioritize")
graph.add_edge("prioritize", "classify")
graph.add_edge("classify", "respond")
graph.add_edge("respond", END)
app = graph.compile()

print(app.get_graph().draw_ascii())


if __name__ == "__main__":
    samples = [
        "My API is down — 500 errors since 7am.",
        "Data sync stuck at 73% for 2 hours.",
        "Can I book a demo next week?",
    ]
    for s in samples:
        r = app.invoke({"ticket_body": s, "priority": None, "category": None, "response": None})
        print(f"\n{s}")
        print(f"  priority -> {r['priority']}")
        print(f"  category -> {r['category']}")
        print(f"  response -> {r['response']}")
