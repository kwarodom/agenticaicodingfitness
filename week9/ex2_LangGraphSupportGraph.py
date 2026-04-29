#ex2_LangGraphSupportGraph.py
import os
from dotenv import load_dotenv
load_dotenv()

'''
Architecture: Graph-Based Routing (Pattern B)
Build a state-machine support ticket routing system using LangGraph. This demonstrates centralized
multi-agent planning with BDI-like deliberation — a router node examines state and selects the appropriate
specialist agent (Wooldridge, 2002; Bellifemine et al., 2007).
'''

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic

class TicketState(TypedDict):
    ticket_text: str
    category: str
    response: str
    confidence: float

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0,
)

def classify_ticket(state: TicketState) -> TicketState:
    """Router node: classifies the ticket category."""
    prompt = f"""Classify this support ticket into exactly one
category: billing, technical, or general.
Ticket: {state["ticket_text"]}
Respond with JSON: {{"category": "...", "confidence": 0.0-1.0}}
"""
    resp = llm.invoke(prompt)
    import json, re
    # Extract JSON from response (model may wrap it in markdown or extra text)
    text = resp.content.strip()
    match = re.search(r'\{[^}]+\}', text)
    if match:
        data = json.loads(match.group())
    else:
        # Fallback: try to infer category from text
        text_lower = text.lower()
        if "billing" in text_lower:
            data = {"category": "billing", "confidence": 0.8}
        elif "technical" in text_lower:
            data = {"category": "technical", "confidence": 0.8}
        else:
            data = {"category": "general", "confidence": 0.5}
    return {
        "category": data["category"],
        "confidence": float(data["confidence"]),
    }

# define specialist agents
def billing_agent(state: TicketState) -> TicketState:
    prompt = f"""You are AltoTech billing support. Help with:
{state["ticket_text"]}
Provide a helpful, professional response."""
    resp = llm.invoke(prompt)
    return {"response": resp.content}

def technical_agent(state: TicketState) -> TicketState:
    prompt = f"""You are AltoTech technical support specializing
in HVAC systems and IoT sensors. Help with:
{state["ticket_text"]}
Provide a detailed technical response."""
    resp = llm.invoke(prompt)
    return {"response": resp.content}

def general_agent(state: TicketState) -> TicketState:
    prompt = f"""You are AltoTech general support. Help with:
{state["ticket_text"]}
Provide a friendly, helpful response."""
    resp = llm.invoke(prompt)
    return {"response": resp.content}

# Step 4: Build the Graph
def route_ticket(state: TicketState) -> str:
    """Conditional edge: routes based on category."""
    return state["category"]

graph = StateGraph(TicketState)
# Add nodes
graph.add_node("classify", classify_ticket)
graph.add_node("billing", billing_agent)
graph.add_node("technical", technical_agent)
graph.add_node("general", general_agent)
# Set entry point
graph.set_entry_point("classify")
# Add conditional routing
graph.add_conditional_edges(
    "classify",
    route_ticket,
    {
        "billing": "billing",
        "technical": "technical",
        "general": "general",
    },
)
# All specialists go to END
graph.add_edge("billing", END)
graph.add_edge("technical", END)
graph.add_edge("general", END)
app = graph.compile()

# Step 5: Test the Graph
# Test with different ticket types
tickets = [
    "My last invoice shows a charge I don't recognize for"
    " sensor maintenance on floor 3.",
    "The CO2 sensor in conference room B is reading 2000ppm"
    " but the room is empty. Possible calibration issue?",
    "Can you send me the brochure for the CERO platform?",
]
for ticket in tickets:
    print(f"\nTicket: {ticket[:60]}...")
    result = app.invoke({"ticket_text": ticket})
    print(f"Category: {result['category']}"
          f" (confidence: {result['confidence']:.0%})")
    print(f"Response: {result['response'][:200]}...")
    print("-" * 60)

#Expected Output
#Each ticket should be classified into the correct category and routed to the appropriate specialist. Notice how
#the state flows through the graph — this is a state machine, analogous to BDI deliberation where the agent
#decides WHAT to do (classify) then HOW to do it (route to specialist).