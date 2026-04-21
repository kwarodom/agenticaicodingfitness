# %% [markdown]
# # Ex 2 SOLUTION — Supervisor + escalation_specialist challenge
#
# Base supervisor + a 4th `escalation_specialist` that fires on cancellation
# signals or profanity.

# %%
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


class SupervisorState(TypedDict):
    ticket_body: str
    messages: Annotated[list, operator.add]
    final_response: str | None


@tool
def technical_specialist(ticket: str) -> str:
    """Handle TECHNICAL support questions (APIs, devices, dashboards, syncs, errors)."""
    return llm.invoke(
        f"You are a senior technical support engineer. Answer crisply:\n\n{ticket}"
    ).content


@tool
def billing_specialist(ticket: str) -> str:
    """Handle BILLING questions (charges, invoices, plans, refunds)."""
    return llm.invoke(
        f"You are a billing specialist. Be precise and reassuring:\n\n{ticket}"
    ).content


@tool
def general_specialist(ticket: str) -> str:
    """Handle GENERAL or pre-sales questions (SLA, references, demos, integrations)."""
    return llm.invoke(
        f"You are a friendly pre-sales specialist. Be helpful and concise:\n\n{ticket}"
    ).content


@tool
def escalation_specialist(ticket: str) -> str:
    """Handle ESCALATIONS: cancellations, refunds with anger, profanity, churn risk.
    Use when the customer asks to cancel their account or uses abusive language."""
    return llm.invoke(
        "You are a retention specialist. The customer is angry or threatening to leave. "
        "Acknowledge their frustration, offer one concrete next step, and promise a human "
        f"follow-up within 1 hour:\n\n{ticket}"
    ).content


SPECIALISTS = [
    technical_specialist,
    billing_specialist,
    general_specialist,
    escalation_specialist,
]

supervisor_prompt = (
    "You are the support supervisor. Read the ticket and pick exactly ONE specialist tool.\n"
    "Route to `escalation_specialist` if the ticket:\n"
    "  - asks to cancel or close the account\n"
    "  - contains profanity or threats\n"
    "  - explicitly says the customer is leaving\n"
    "Otherwise pick technical / billing / general based on topic. "
    "Call the tool with the original ticket, then stop."
)

supervisor = create_react_agent(
    model=llm,
    tools=SPECIALISTS,
    prompt=supervisor_prompt,
)


def run_supervisor(state: SupervisorState) -> SupervisorState:
    result = supervisor.invoke({"messages": [("user", state["ticket_body"])]})
    return {
        "final_response": result["messages"][-1].content,
        "messages": result["messages"],
    }


graph = StateGraph(SupervisorState)
graph.add_node("supervisor", run_supervisor)
graph.add_edge(START, "supervisor")
graph.add_edge("supervisor", END)
app = graph.compile()


if __name__ == "__main__":
    samples = [
        "My dashboard won't load.",
        "I was charged twice.",
        "Do you have a retail reference?",
        "Cancel my account immediately, I'm done with you.",
    ]
    for s in samples:
        out = app.invoke({"ticket_body": s, "messages": [], "final_response": None})
        print("\n---")
        print(f"Ticket:  {s}")
        print(f"Answer:  {out['final_response'][:220]}...")
