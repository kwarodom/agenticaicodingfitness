"""
Exercise 05 — LangGraph FAQ Loop with Memory (Easy).

A multi-turn FAQ agent using LangGraph's message accumulator pattern. State
carries the full conversation history via Annotated[list, operator.add]. A
conditional edge decides whether to continue the loop or terminate.

Key concepts:
    - Annotated[list, operator.add] for append-only message history
    - Cycles in a graph — a node can route back to itself
    - Termination condition via conditional edge
    - SystemMessage + HumanMessage + AIMessage from langchain_core
"""

import operator
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

SYSTEM = (
    "You are CERO Assistant for AltoTech smart building platform. "
    "Help hotel staff with: energy settings, HVAC alerts, schedules. "
    "Keep answers to 2-3 sentences. "
    "If the issue is fully resolved, end with RESOLVED: YES."
)


class FAQState(TypedDict):
    messages: Annotated[list, operator.add]
    turns: int
    resolved: bool


def faq_node(state: FAQState) -> FAQState:
    msgs = [SystemMessage(content=SYSTEM)] + state["messages"]
    reply = llm.invoke(msgs)
    resolved = "resolved: yes" in reply.content.lower()
    print(f"Assistant: {reply.content[:120]}...")
    return {
        "messages": [AIMessage(content=reply.content)],
        "turns": state["turns"] + 1,
        "resolved": resolved,
    }


def user_node(state: FAQState) -> FAQState:
    follow_ups = [
        "Which zones use the most power right now?",
        "How do I reset the night-mode schedule?",
        "Can I get a weekly energy report emailed to me?",
    ]
    idx = state["turns"]
    if idx >= len(follow_ups):
        return {"resolved": True}
    msg = follow_ups[idx]
    print(f"\nUser: {msg}")
    return {"messages": [HumanMessage(content=msg)]}


def should_continue(state: FAQState) -> str:
    if state["resolved"] or state["turns"] >= 4:
        return "end"
    return "user"


builder = StateGraph(FAQState)
builder.add_node("faq", faq_node)
builder.add_node("user", user_node)
builder.add_edge(START, "faq")
builder.add_edge("user", "faq")
builder.add_conditional_edges(
    "faq", should_continue, {"user": "user", "end": END}
)
graph = builder.compile()


if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph multi-turn FAQ loop")
    print("=" * 60)

    initial_user = "My hotel's HVAC is using 20% more energy. What's wrong?"
    print(f"\nUser: {initial_user}")
    result = graph.invoke({
        "messages": [HumanMessage(content=initial_user)],
        "turns": 0,
        "resolved": False,
    })
    print(
        f"\nSession done. Turns: {result['turns']}, "
        f"Resolved: {result['resolved']}"
    )
