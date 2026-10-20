"""
Exercise 03 — LangGraph Query Router (Beginner).

A TypedDict state flows through a classifier node, then branches via
add_conditional_edges to one of three specialist nodes. Demonstrates the
fundamental LangGraph compile -> invoke pattern.

Key concepts:
    - TypedDict state schema
    - StateGraph, add_node, add_edge, add_conditional_edges
    - START and END sentinels
    - Routing functions that return node-name strings
"""

import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic

load_dotenv()

os.environ.setdefault("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")


class State(TypedDict):
    query: str
    category: str
    response: str


def classify(state: State) -> State:
    result = llm.invoke(
        f"Classify into ONE word — billing, technical, or general.\n"
        f"Query: {state['query']}\nReply with one word only."
    )
    return {"category": result.content.strip().lower()}


def billing_agent(state: State) -> State:
    r = llm.invoke(
        f"You are a billing specialist. Answer: {state['query']}"
    )
    return {"response": f"[Billing] {r.content}"}


def tech_agent(state: State) -> State:
    r = llm.invoke(
        f"You are a tech support engineer. Answer: {state['query']}"
    )
    return {"response": f"[Tech] {r.content}"}


def general_agent(state: State) -> State:
    r = llm.invoke(f"You are a helpful agent. Answer: {state['query']}")
    return {"response": f"[General] {r.content}"}


def router(state: State) -> Literal["billing_agent", "tech_agent", "general_agent"]:
    c = state["category"]
    if "billing" in c:
        return "billing_agent"
    if "technical" in c or "tech" in c:
        return "tech_agent"
    return "general_agent"


builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("billing_agent", billing_agent)
builder.add_node("tech_agent", tech_agent)
builder.add_node("general_agent", general_agent)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", router)
builder.add_edge("billing_agent", END)
builder.add_edge("tech_agent", END)
builder.add_edge("general_agent", END)

graph = builder.compile()


QUERIES = [
    "Why was I charged twice this month?",
    "My smart thermostat won't connect to the app.",
    "What are your support hours?",
]


if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph query router")
    print("=" * 60)

    for q in QUERIES:
        res = graph.invoke({"query": q, "category": "", "response": ""})
        print(f"\nQ: {q}")
        print(f"Category: {res['category']}")
        print(f"A: {res['response'][:200]}...\n")
