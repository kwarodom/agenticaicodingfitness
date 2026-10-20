#!/usr/bin/env python3
"""Part 3.2 — LangGraph ReAct agent over the graph (kg_mastery.pdf §3.2).

KEY INSIGHT: GraphCypherQAChain does ONE hop (question → Cypher → answer). A
ReAct agent can REASON and ACT in a loop: it decides which tool to call, reads
the result, and decides what to do next — and it can WRITE back to the graph.
Here the graph is both the agent's knowledge source AND its long-term memory:
the agent stores key findings as :Event nodes it can recall later.

Tools:
  query_knowledge_graph(cypher)            → read the hotel graph
  store_agent_event(agent_id, type, details) → persist a finding as an Event

Demo: "Which rooms on floor 3 have had HVAC issues in the last 7 days?"

Requires:
  - ANTHROPIC_API_KEY in your environment / repo-root .env
  - pip install langgraph langchain-anthropic langchain-core

Run:  python week15/kg_mastery/part3_graphrag/02_langgraph_react_agent.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import LLM_MODEL, check_connection, get_driver

try:
    from langgraph.prebuilt import create_react_agent
    from langchain_core.tools import tool
    from langchain_anthropic import ChatAnthropic
except ImportError as e:
    print(f"⚠️  Missing dependency: {e.name}")
    print("   pip install langgraph langchain-anthropic langchain-core")
    sys.exit(1)


@tool
def query_knowledge_graph(cypher: str) -> str:
    """Execute a Cypher query on the hotel knowledge graph.

    Use for room status, device readings, incident history. Pass a single
    valid Cypher statement. Returns up to 50 rows as a JSON string.
    """
    driver = get_driver()
    try:
        with driver.session() as s:
            rows = [r.data() for r in s.run(cypher)][:50]
        return json.dumps(rows, default=str)
    except Exception as e:  # noqa: BLE001
        return f"ERROR running Cypher: {type(e).__name__}: {e}"
    finally:
        driver.close()


@tool
def store_agent_event(agent_id: str, event_type: str, details: str) -> str:
    """Store a key finding as an Event in the knowledge graph (the agent's memory).

    Call this when you discover something worth remembering, e.g. a confirmed
    incident. agent_id identifies you; event_type is a short label such as
    'ANALYSIS' or 'INCIDENT'; details is a human-readable summary string.
    """
    driver = get_driver()
    try:
        with driver.session() as s:
            s.run(
                """
                MERGE (a:Agent {id: $agent_id})
                CREATE (e:Event {
                    id: randomUUID(),
                    type: $event_type,
                    details: $details,
                    ts: datetime()
                })
                CREATE (a)-[:PERFORMED]->(e)
                """,
                agent_id=agent_id, event_type=event_type, details=details,
            )
        return f"Stored {event_type} event for agent {agent_id}."
    except Exception as e:  # noqa: BLE001
        return f"ERROR storing event: {type(e).__name__}: {e}"
    finally:
        driver.close()


SYSTEM_PROMPT = (
    "You are a hotel energy-management AI. You have a knowledge graph of rooms, "
    "devices, sensor readings, alerts and maintenance jobs. Use "
    "query_knowledge_graph to investigate questions with Cypher. When you reach "
    "a meaningful conclusion, persist it with store_agent_event (agent_id "
    "'energy_mgmt_v1') so you can recall it later. Be concise and cite the data."
)


def build_agent():
    llm = ChatAnthropic(model=LLM_MODEL, temperature=0)
    return create_react_agent(
        llm,
        tools=[query_knowledge_graph, store_agent_event],
        prompt=SYSTEM_PROMPT,
    )


def main():
    agent = build_agent()
    question = (
        "Which rooms on floor 3 have had HVAC issues in the last 7 days?"
    )
    print(f"=== Q: {question} ===\n")
    result = agent.invoke({"messages": [("user", question)]})
    last = result["messages"][-1]
    print("Final answer:")
    print(last.content)


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set.")
        print("   export ANTHROPIC_API_KEY=sk-ant-...  (or add it to repo-root .env)")
        sys.exit(1)
    if not check_connection():
        sys.exit(1)
    main()
