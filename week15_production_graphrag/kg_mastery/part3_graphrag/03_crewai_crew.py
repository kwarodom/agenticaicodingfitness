#!/usr/bin/env python3
"""Part 3.3 — CrewAI multi-agent crew over the graph (kg_mastery.pdf §3.3).

KEY INSIGHT: CrewAI splits work across ROLE-SPECIALISED agents that hand results
to each other. We give each agent graph-backed tools so the knowledge graph is
the shared substrate the whole crew reads from and writes to:

  Analyst   (Hotel Energy Analyst)  — reads the graph (KnowledgeGraphTool)
  Optimizer (Energy Optimizer)      — reads + writes recommendations back
                                       (KnowledgeGraphTool + GraphMemoryTool)

Two sequential tasks: (1) analyse high-consumption rooms, then
(2) recommend optimisations and persist them as :OptimizationAction nodes.

NOTE: CrewAI takes the model as a string. For Anthropic, prefix the model id
with 'anthropic/', i.e. f"anthropic/{LLM_MODEL}".

Requires:
  - ANTHROPIC_API_KEY in your environment / repo-root .env
  - pip install crewai crewai-tools

Run:  python week15/kg_mastery/part3_graphrag/03_crewai_crew.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import LLM_MODEL, check_connection, get_driver

try:
    from crewai import Agent, Task, Crew, Process
    from crewai_tools import BaseTool
except ImportError as e:
    print(f"⚠️  Missing dependency: {e.name}")
    print("   pip install crewai crewai-tools")
    sys.exit(1)


class KnowledgeGraphTool(BaseTool):
    name: str = "knowledge_graph_query"
    description: str = (
        "Run a read-only Cypher query against the hotel knowledge graph "
        "(rooms, devices, sensor readings, alerts, maintenance). Input is a "
        "single Cypher statement string; returns rows as JSON."
    )

    def _run(self, cypher: str) -> str:
        driver = get_driver()
        try:
            with driver.session() as s:
                rows = [r.data() for r in s.run(cypher)][:50]
            return json.dumps(rows, default=str)
        except Exception as e:  # noqa: BLE001
            return f"ERROR running Cypher: {type(e).__name__}: {e}"
        finally:
            driver.close()


class GraphMemoryTool(BaseTool):
    name: str = "store_to_graph"
    description: str = (
        "Persist a node to the knowledge graph. Inputs: label (e.g. "
        "'OptimizationAction'), properties (a JSON object string of node "
        "properties), and optional relationship (currently informational). "
        "MERGEs the node so repeated calls are idempotent on its properties."
    )

    def _run(self, label: str, properties: str, relationship: str = "") -> str:
        try:
            props = properties if isinstance(properties, dict) else json.loads(properties)
        except (json.JSONDecodeError, TypeError) as e:
            return f"ERROR: properties must be a JSON object: {e}"
        driver = get_driver()
        try:
            with driver.session() as s:
                s.run(
                    f"MERGE (n:`{label}` $props) SET n.created_at = datetime()",
                    props=props,
                )
            return f"Stored :{label} node with properties {props}."
        except Exception as e:  # noqa: BLE001
            return f"ERROR storing node: {type(e).__name__}: {e}"
        finally:
            driver.close()


def build_crew():
    model = f"anthropic/{LLM_MODEL}"  # CrewAI provider-prefixed model string
    kg_tool = KnowledgeGraphTool()
    mem_tool = GraphMemoryTool()

    analyst = Agent(
        role="Hotel Energy Analyst",
        goal="Identify the highest energy-consuming rooms and why they run hot.",
        backstory=(
            "A meticulous building-analytics specialist who lives in the sensor "
            "data and never guesses without querying the graph."
        ),
        tools=[kg_tool],
        llm=model,
        verbose=True,
    )

    optimizer = Agent(
        role="Energy Optimizer",
        goal=(
            "Turn the analyst's findings into concrete optimisation actions and "
            "record them in the knowledge graph."
        ),
        backstory=(
            "A pragmatic facilities engineer who recommends specific, "
            "low-risk HVAC and scheduling changes and logs every decision."
        ),
        tools=[kg_tool, mem_tool],
        llm=model,
        verbose=True,
    )

    analyse_task = Task(
        description=(
            "Query the knowledge graph for rooms with the highest recent energy "
            "consumption and elevated temperatures. Summarise the top offenders "
            "with their floor, energy_kwh and temperature evidence."
        ),
        expected_output="A ranked list of high-consumption rooms with evidence.",
        agent=analyst,
    )

    optimise_task = Task(
        description=(
            "Using the analyst's findings, recommend a specific optimisation for "
            "each high-consumption room. For each recommendation, store an "
            ":OptimizationAction node (properties: room, action, expected_saving_kwh) "
            "via the store_to_graph tool."
        ),
        expected_output=(
            "A list of recommended OptimizationActions, confirmed as stored."
        ),
        agent=optimizer,
        context=[analyse_task],
    )

    return Crew(
        agents=[analyst, optimizer],
        tasks=[analyse_task, optimise_task],
        process=Process.sequential,
        verbose=True,
    )


def main():
    crew = build_crew()
    result = crew.kickoff()
    print("\n=== Crew result ===")
    print(result)


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set.")
        print("   export ANTHROPIC_API_KEY=sk-ant-...  (or add it to repo-root .env)")
        sys.exit(1)
    if not check_connection():
        sys.exit(1)
    main()
