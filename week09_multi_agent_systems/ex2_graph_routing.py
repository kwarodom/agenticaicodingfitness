"""
Exercise 2 — Graph-Based Routing (Pattern B) with LangGraph.

A classifier node inspects each support ticket, decides its category, and a
conditional edge routes it to one of three specialist nodes (billing,
technical, general). The shared TicketState TypedDict is the modern
equivalent of the FIPA blackboard / shared beliefs pattern.

Maps to: centralised multi-agent planning + BDI deliberation
(Wooldridge 2002 Ch.4; Bellifemine et al. 2007).
    classify -> beliefs + intention selection
    specialist -> plan execution
"""

import json
import os
import re
from typing import TypedDict
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
llm = ChatAnthropic(
    model=MODEL,
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0,
)


class TicketState(TypedDict):
    ticket_text: str
    category: str
    confidence: float
    response: str


def classify_ticket(state: TicketState) -> TicketState:
    """Router node — classifies ticket and emits a confidence score."""
    prompt = (
        "Classify this support ticket into exactly one category: "
        "billing, technical, or general.\n\n"
        f"Ticket: {state['ticket_text']}\n\n"
        'Respond with JSON only: {"category": "...", "confidence": 0.0-1.0}'
    )
    resp = llm.invoke(prompt)
    text = resp.content.strip()

    match = re.search(r"\{[^}]+\}", text)
    if match:
        data = json.loads(match.group())
    else:
        # Fallback if the model didn't emit JSON.
        lowered = text.lower()
        if "billing" in lowered:
            data = {"category": "billing", "confidence": 0.7}
        elif "technical" in lowered:
            data = {"category": "technical", "confidence": 0.7}
        else:
            data = {"category": "general", "confidence": 0.5}

    return {
        "category": data["category"],
        "confidence": float(data["confidence"]),
    }


def billing_agent(state: TicketState) -> TicketState:
    prompt = (
        "You are AltoTech billing support. Help the customer with this "
        f"issue:\n\n{state['ticket_text']}\n\n"
        "Provide a helpful, professional response."
    )
    return {"response": llm.invoke(prompt).content}


def technical_agent(state: TicketState) -> TicketState:
    prompt = (
        "You are AltoTech technical support specialising in HVAC systems "
        f"and IoT sensors. Help with:\n\n{state['ticket_text']}\n\n"
        "Provide a detailed technical response."
    )
    return {"response": llm.invoke(prompt).content}


def general_agent(state: TicketState) -> TicketState:
    prompt = (
        "You are AltoTech general support. Help with:\n\n"
        f"{state['ticket_text']}\n\n"
        "Provide a friendly, helpful response."
    )
    return {"response": llm.invoke(prompt).content}


def route_ticket(state: TicketState) -> str:
    """Conditional edge — picks the next node based on classifier output."""
    return state["category"]


def build_graph():
    graph = StateGraph(TicketState)
    graph.add_node("classify", classify_ticket)
    graph.add_node("billing", billing_agent)
    graph.add_node("technical", technical_agent)
    graph.add_node("general", general_agent)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_ticket,
        {
            "billing": "billing",
            "technical": "technical",
            "general": "general",
        },
    )
    graph.add_edge("billing", END)
    graph.add_edge("technical", END)
    graph.add_edge("general", END)

    return graph.compile()


TICKETS = [
    "My last invoice shows a charge I don't recognise for sensor "
    "maintenance on floor 3.",
    "The CO2 sensor in conference room B is reading 2000ppm but the room "
    "is empty. Possible calibration issue?",
    "Can you send me the brochure for the CERO platform?",
]


if __name__ == "__main__":
    app = build_graph()

    print("=" * 60)
    print("LangGraph state-machine routing: classify -> specialist")
    print("=" * 60)

    transcripts = []
    for i, ticket in enumerate(TICKETS, 1):
        print(f"\n[{i}/{len(TICKETS)}] Ticket: {ticket[:70]}...")
        result = app.invoke({"ticket_text": ticket})
        print(
            f"      routed to: {result['category']:<10} "
            f"(confidence {result['confidence']:.0%})"
        )
        print(f"      response : {result['response'][:140]}...")
        transcripts.append(
            {
                "ticket": ticket,
                "category": result["category"],
                "confidence": result["confidence"],
                "response": result["response"],
            }
        )

    out = os.path.join(os.path.dirname(__file__), "ex2_routing_output.md")
    with open(out, "w") as f:
        f.write("# Exercise 2 — Graph Routing Output\n\n")
        for t in transcripts:
            f.write(f"## Ticket\n{t['ticket']}\n\n")
            f.write(
                f"**Category:** {t['category']} "
                f"(confidence {t['confidence']:.0%})\n\n"
            )
            f.write(f"**Response:**\n\n{t['response']}\n\n---\n\n")

    print(f"\nFull transcripts saved to: {out}")
