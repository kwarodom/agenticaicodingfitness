#!/usr/bin/env python3
"""Part 5.1 — Hotel IoT operations agent, full reference (kg_mastery.pdf §5.1).

The flagship real-world use case: a LangGraph ReAct agent that monitors hotel
room conditions and coordinates maintenance over the LIVE hotel knowledge graph
(the one already loaded by Part 1/2). It runs a reason→act loop, picking tools,
reading their results, and finally WRITING a maintenance assignment back into
the graph.

Three @tool functions wrap get_driver(); each holds its own Cypher:

  get_hot_rooms(threshold_c)      → rooms with a recent SensorReading over threshold
  get_unresolved_alerts(severity) → open alerts at a severity, joined back to Room
  assign_maintenance(room, staff, → create a MaintenanceJob + wire it to staff/room
                      job_type)

KEY INSIGHT (from the PDF): tool NAMING and DESCRIPTIONS are the agent's whole
interface — name them the way you'd name an API. Keep the Cypher INSIDE the tool,
hidden from the model: the agent reasons in plain language ("find hot rooms",
"assign maintenance") and never has to write a query. This is safer (no arbitrary
Cypher), cheaper (smaller prompts), and far more reliable than NL-to-Cypher.

Requires:
  - ANTHROPIC_API_KEY in your environment / repo-root .env
  - pip install langgraph langchain-anthropic langchain-core

Run:  python week15/kg_mastery/part5_use_cases/01_hotel_iot_agent.py
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
def get_hot_rooms(threshold_c: float = 28.0) -> str:
    """List rooms running hot: any recent SensorReading with temp_c above threshold_c.

    Use this first to find rooms that may need HVAC attention. threshold_c is the
    temperature in Celsius above which a room counts as "hot" (default 28.0).
    Returns JSON rows of room id, floor, and the room's max recent temperature.
    """
    # Spec says "last 30 min", but our sample data is hourly — widen to 24h so
    # the demo returns rows. Swap to duration({minutes: 30}) on live streams.
    driver = get_driver()
    try:
        with driver.session() as s:
            rows = [
                r.data()
                for r in s.run(
                    """
                    MATCH (room:Room)-[:HAS_READING]->(sr:SensorReading)
                    WHERE sr.temp_c > $threshold
                      AND sr.ts >= datetime() - duration({hours: 24})
                    RETURN room.id AS room, room.floor AS floor,
                           max(sr.temp_c) AS max_temp_c
                    ORDER BY max_temp_c DESC
                    """,
                    threshold=threshold_c,
                )
            ]
        return json.dumps(rows, default=str) if rows else "No hot rooms found."
    except Exception as e:  # noqa: BLE001
        return f"ERROR in get_hot_rooms: {type(e).__name__}: {e}"
    finally:
        driver.close()


@tool
def get_unresolved_alerts(severity: str = "HIGH") -> str:
    """List unresolved alerts at a given severity, joined back to the affected room.

    Use this to find problems that still need a maintenance job. severity is one
    of 'LOW', 'MEDIUM', 'HIGH' (default 'HIGH'). Returns JSON rows of alert id,
    type, message, the device that triggered it, and the room it belongs to.
    """
    driver = get_driver()
    try:
        with driver.session() as s:
            rows = [
                r.data()
                for r in s.run(
                    """
                    MATCH (room:Room)-[:HAS_DEVICE]->(dev:Device)-[:TRIGGERED]->(a:Alert)
                    WHERE a.resolved = false AND a.severity = $sev
                    RETURN a.id AS alert, a.type AS type, a.message AS message,
                           a.severity AS severity, dev.id AS device, room.id AS room
                    ORDER BY a.ts DESC
                    """,
                    sev=severity,
                )
            ]
        return json.dumps(rows, default=str) if rows else f"No unresolved {severity} alerts."
    except Exception as e:  # noqa: BLE001
        return f"ERROR in get_unresolved_alerts: {type(e).__name__}: {e}"
    finally:
        driver.close()


@tool
def assign_maintenance(room_id: str, staff_id: str, job_type: str) -> str:
    """Create and assign a maintenance job for a room to a staff member.

    Use this once you've decided a room needs work. room_id e.g. 'R101';
    staff_id e.g. 'S1'; job_type a short label such as 'HVAC_REPAIR' or
    'INSPECTION'. Creates a MaintenanceJob (status ASSIGNED) and links it to
    both the staff member and the room. Returns the new job id.
    """
    driver = get_driver()
    try:
        with driver.session() as s:
            rec = s.run(
                """
                MATCH (room:Room {id: $room_id})
                MATCH (staff:Staff {id: $staff_id})
                CREATE (job:MaintenanceJob {
                    id: randomUUID(),
                    type: $job_type,
                    status: 'ASSIGNED',
                    started_at: datetime()
                })
                CREATE (staff)-[:PERFORMED]->(job)
                CREATE (job)-[:FOR_ROOM]->(room)
                RETURN job.id AS job_id
                """,
                room_id=room_id, staff_id=staff_id, job_type=job_type,
            ).single()
        if rec is None:
            return (
                f"Could not assign: check that room '{room_id}' and staff "
                f"'{staff_id}' both exist."
            )
        return f"Assigned {job_type} job {rec['job_id']} for room {room_id} to staff {staff_id}."
    except Exception as e:  # noqa: BLE001
        return f"ERROR in assign_maintenance: {type(e).__name__}: {e}"
    finally:
        driver.close()


SYSTEM_PROMPT = (
    "You are a hotel operations AI. Use the knowledge graph to monitor room "
    "conditions and coordinate maintenance. Always check for hot rooms and "
    "unresolved HIGH alerts first."
)


def build_agent():
    llm = ChatAnthropic(model=LLM_MODEL, temperature=0)
    return create_react_agent(
        llm,
        tools=[get_hot_rooms, get_unresolved_alerts, assign_maintenance],
        prompt=SYSTEM_PROMPT,
    )


def main():
    agent = build_agent()
    question = (
        "Do a morning ops check: which rooms are running hot and what HIGH "
        "alerts are still open? If a room is both hot and has an open HVAC "
        "alert, assign an HVAC_REPAIR job for it to staff S1."
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
