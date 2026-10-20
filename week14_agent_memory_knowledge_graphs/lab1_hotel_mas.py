"""lab1_hotel_mas.py — Hotel Multi-Agent System (CrewAI + Neo4j knowledge graph).

A 2-agent crew that optimises hotel HVAC energy use:

    Sensor Monitor  --(flagged hot zones)-->  Energy Optimizer

Both agents act on the LIVE knowledge graph built by 6.hotel_kg_builder.py.
Every decision the Optimizer makes is persisted back into the graph as an
Event node via AgentMemory — so the MAS has durable, queryable memory.

Run:
    ./.venv/bin/python week14/lab1_hotel_mas.py        # from repo root
    # or, from inside week14/:
    ../.venv/bin/python lab1_hotel_mas.py

Prerequisites:
    - Neo4j running at bolt://localhost:7687 (docker compose up -d)
    - Hotel graph built:  python week14/6.hotel_kg_builder.py
    - ANTHROPIC_API_KEY in the repo-root .env
"""
import os
import json

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from neo4j import GraphDatabase

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# AgentMemory lives next to this file; make sure it is importable.
import sys
sys.path.insert(0, os.path.dirname(__file__))
from agent_memory import AgentMemory, URI, AUTH

# Load ANTHROPIC_API_KEY from the repo-root .env (one level up from week14/).
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Shared infrastructure ──────────────────────────────────────────────────────
# One driver for tool reads/writes; one AgentMemory so the Optimizer can log
# its decisions into the same graph the hotel lives in.
DRIVER = GraphDatabase.driver(URI, auth=AUTH)
OPTIMIZER_MEMORY = AgentMemory("HVACControlAgent")  # reuse the KG's existing agent node

# Anthropic LLM for both agents (CrewAI routes through litellm → "anthropic/<model>").
LLM_MODEL = LLM(
    model="anthropic/claude-haiku-4-5-20251001",
    temperature=0.1,
    api_key=os.environ["ANTHROPIC_API_KEY"],
)


# ── Tool 1: scan the graph for zones that need cooling ──────────────────────────
class ScanArgs(BaseModel):
    over_by_c: float = Field(
        1.0, description="Flag zones whose temperature exceeds setpoint by at least this many °C."
    )


class ScanZonesTool(BaseTool):
    name: str = "scan_hot_zones"
    description: str = (
        "Scan the hotel knowledge graph and return zones whose current temperature "
        "is above their HVAC setpoint. Use this to find rooms/zones that need cooling."
    )
    args_schema: type[BaseModel] = ScanArgs

    def _run(self, over_by_c: float = 1.0) -> str:
        cypher = """
            MATCH (z:Zone)-[:HAS_HVAC]->(u:HVACUnit)
            WHERE z.current_temp_c >= z.setpoint_celsius + $over_by_c
            RETURN z.id              AS zone_id,
                   z.type            AS zone_type,
                   z.current_temp_c  AS current_temp,
                   z.setpoint_celsius AS setpoint,
                   round(z.current_temp_c - z.setpoint_celsius, 2) AS over_by,
                   u.id              AS hvac_unit,
                   u.power_kw        AS power_kw
            ORDER BY over_by DESC
            LIMIT 8
        """
        with DRIVER.session() as s:
            rows = [dict(r) for r in s.run(cypher, over_by_c=over_by_c)]
        return json.dumps(rows, indent=2)


# ── Tool 2: act — lower a zone's HVAC setpoint and remember the decision ────────
class SetpointArgs(BaseModel):
    zone_id: str = Field(..., description="Zone id, e.g. 'zone-FLOOR-07'.")
    reduce_by_c: float = Field(
        1.0, description="How many °C to lower the HVAC setpoint (more cooling)."
    )


class SetSetpointTool(BaseTool):
    name: str = "lower_hvac_setpoint"
    description: str = (
        "Lower the HVAC setpoint for a zone to increase cooling. "
        "Writes the change to the knowledge graph and records the decision in agent memory. "
        "Returns the old and new setpoint."
    )
    args_schema: type[BaseModel] = SetpointArgs

    def _run(self, zone_id: str, reduce_by_c: float = 1.0) -> str:
        cypher = """
            MATCH (z:Zone {id: $zone_id})-[:HAS_HVAC]->(u:HVACUnit)
            WITH z, u, u.setpoint_celsius AS old_sp
            SET u.setpoint_celsius = old_sp - $reduce_by_c,
                u.last_adjusted    = datetime()
            RETURN z.id AS zone_id, old_sp AS old_setpoint,
                   u.id AS hvac_unit, u.setpoint_celsius AS new_setpoint
        """
        with DRIVER.session() as s:
            rec = s.run(cypher, zone_id=zone_id, reduce_by_c=reduce_by_c).single()

        if rec is None:
            return f"ERROR: no HVAC unit found for zone '{zone_id}'."

        result = dict(rec)
        # Persist the decision as an Event linked to the agent (MAS memory).
        OPTIMIZER_MEMORY.store_event(
            "SETPOINT_ADJUSTED",
            {
                "zone": result["zone_id"],
                "old_setpoint": result["old_setpoint"],
                "new_setpoint": result["new_setpoint"],
                "reason": "zone_over_temperature",
            },
            entities=[("Zone", result["zone_id"]), ("HVACUnit", result["hvac_unit"])],
        )
        return json.dumps(result)


# ── Agents ──────────────────────────────────────────────────────────────────────
sensor_agent = Agent(
    role="Sensor Monitor",
    goal="Detect hotel zones that are above their cooling setpoint",
    backstory="An IoT specialist watching 30 zones across the Grand Vista Hotel.",
    tools=[ScanZonesTool()],
    llm=LLM_MODEL,
    verbose=True,
    max_iter=4,
)

optimizer = Agent(
    role="Energy Optimizer",
    goal="Cool over-temperature zones with the smallest energy penalty, and log every action",
    backstory="An energy engineer who tunes HVAC setpoints and keeps an audit trail.",
    tools=[SetSetpointTool()],
    llm=LLM_MODEL,
    verbose=True,
    max_iter=6,
)


# ── Tasks (sequential pipeline) ──────────────────────────────────────────────────
scan = Task(
    description=(
        "Use the scan_hot_zones tool (over_by_c=1.0) to list every hotel zone whose "
        "temperature is at least 1°C above its setpoint. Report the zone_id, current_temp, "
        "setpoint and over_by for each."
    ),
    expected_output="A JSON list of flagged zones with their temperature readings.",
    agent=sensor_agent,
)

optimize = Task(
    description=(
        "For each flagged zone from the previous step, call lower_hvac_setpoint with "
        "reduce_by_c=1.0 to increase cooling. After adjusting all zones, summarise the "
        "actions: zone_id, old_setpoint -> new_setpoint."
    ),
    expected_output="A summary table of every zone adjusted and its old/new setpoint.",
    agent=optimizer,
    context=[scan],
)


# ── Crew ──────────────────────────────────────────────────────────────────────────
crew = Crew(
    agents=[sensor_agent, optimizer],
    tasks=[scan, optimize],
    process=Process.sequential,
    memory=False,          # disable CrewAI's built-in (OpenAI-embedding) memory
    verbose=True,
)


if __name__ == "__main__":
    print("=" * 64)
    print("Hotel MAS — Sensor Monitor  ->  Energy Optimizer  (CrewAI + Neo4j)")
    print("=" * 64)

    result = crew.kickoff()

    print("\n" + "=" * 64)
    print("CREW RESULT")
    print("=" * 64)
    print(result)

    # Show the durable memory the MAS just wrote into the graph.
    print("\n" + "=" * 64)
    print("AGENT MEMORY — last 5 SETPOINT_ADJUSTED events in the graph")
    print("=" * 64)
    for e in OPTIMIZER_MEMORY.recall_recent(event_type="SETPOINT_ADJUSTED", limit=5):
        print(f"  [{e['type']}] {e['data']}")

    OPTIMIZER_MEMORY.close()
    DRIVER.close()
    print("\nDone.")
