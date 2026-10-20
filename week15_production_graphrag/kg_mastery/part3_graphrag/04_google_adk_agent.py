#!/usr/bin/env python3
"""Part 3.4 — Google ADK agent over the graph (kg_mastery.pdf §3.4).

KEY INSIGHT: Google's Agent Development Kit (ADK) wraps PLAIN Python functions as
tools — the docstring and type hints ARE the tool schema the model reads, so
write them well. Here two graph-backed functions become the agent's tools, and
a Runner + InMemorySessionService give it a conversation loop.

This is a CONSTRUCTION demo: it builds and wires the agent but does not require a
live Gemini call (ADK uses Gemini, which needs Google credentials — either a
GOOGLE_API_KEY or Vertex AI / application-default credentials).

Tools:
  query_hotel_graph(cypher)                          → run Cypher, return rows
  find_related_entities(entity_type, entity_name, relationship, hops) → neighbours

Requires:
  - pip install google-adk
  - Google credentials for an actual run (GOOGLE_API_KEY or Vertex AI ADC)

Run:  python week15/kg_mastery/part3_graphrag/04_google_adk_agent.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import check_connection, get_driver

try:
    from google.adk.agents import Agent
    from google.adk.tools import FunctionTool
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
except ImportError as e:
    print(f"⚠️  Missing dependency: {e.name}")
    print("   pip install google-adk")
    sys.exit(1)


def query_hotel_graph(cypher: str) -> dict:
    """Run a read-only Cypher query against the hotel knowledge graph.

    Use this for room status, device readings, alerts, and maintenance history.

    Args:
        cypher: A single valid Cypher statement to execute.

    Returns:
        A dict with 'rows' (a list of result records, max 50) on success, or
        'error' (a description string) on failure.
    """
    driver = get_driver()
    try:
        with driver.session() as s:
            rows = [r.data() for r in s.run(cypher)][:50]
        return {"rows": json.loads(json.dumps(rows, default=str))}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        driver.close()


def find_related_entities(
    entity_type: str, entity_name: str, relationship: str, hops: int = 1
) -> dict:
    """Find entities related to a named node via a relationship, up to N hops.

    Args:
        entity_type: The starting node label, e.g. 'Room' or 'Device'.
        entity_name: The starting node's id property value, e.g. 'R305'.
        relationship: The relationship type to traverse, e.g. 'HAS_DEVICE'.
        hops: Maximum traversal depth (default 1).

    Returns:
        A dict with 'neighbours' (a list of related node records) on success, or
        'error' (a description string) on failure.
    """
    driver = get_driver()
    try:
        cypher = (
            f"MATCH (start:`{entity_type}` {{id: $name}})"
            f"-[:`{relationship}`*1..{int(hops)}]-(related) "
            f"RETURN DISTINCT labels(related) AS labels, related.id AS id, "
            f"properties(related) AS props LIMIT 50"
        )
        with driver.session() as s:
            rows = [r.data() for r in s.run(cypher, name=entity_name)]
        return {"neighbours": json.loads(json.dumps(rows, default=str))}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        driver.close()


def build_agent_and_runner():
    agent = Agent(
        name="hotel_energy_agent",
        model="gemini-2.0-flash",
        description="Answers questions about the hotel knowledge graph.",
        instruction=(
            "You are a hotel energy-management assistant. Use query_hotel_graph "
            "to run Cypher and find_related_entities to explore neighbours. "
            "Always ground your answers in the graph data."
        ),
        tools=[
            FunctionTool(query_hotel_graph),
            FunctionTool(find_related_entities),
        ],
    )

    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="hotel_energy_app",
        session_service=session_service,
    )
    return agent, runner


def main():
    agent, runner = build_agent_and_runner()
    print("✅ Google ADK agent constructed and wired to the Neo4j graph.")
    print(f"   agent.name  = {agent.name}")
    print(f"   agent.model = {agent.model}")
    print(f"   tools       = {[t.name for t in agent.tools]}")
    print(f"   runner      = {type(runner).__name__} (InMemorySessionService)")
    print("\nExample query you would send to the runner:")
    print("   'Which devices are attached to room R305, and have any of them "
          "triggered HIGH alerts?'")
    print("\nNOTE: an actual run needs Google credentials (GOOGLE_API_KEY or "
          "Vertex AI ADC). This script is a construction demo and does not call "
          "Gemini.")


if __name__ == "__main__":
    if not check_connection():
        sys.exit(1)
    main()
