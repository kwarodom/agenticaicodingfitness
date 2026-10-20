#!/usr/bin/env python3
"""Lesson 07.1 — Durable, structured agent memory in Neo4j.

Prereq: a running Neo4j. Start one with:
    docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
Then open the browser UI at http://localhost:7474 to WATCH nodes appear.

Run:  python week15/code/07_agent_memory/01_memory_demo.py

What you'll see:
  - an HVAC agent stores 3 events (fault → work order → resolved)
  - we recall recent events, and everything involving Room 301
  - a SECOND agent writes its own events to the SAME graph
  - rerun the script: the memory is still there (unlike folder 01's messages list)
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from agent_memory import AgentMemory, URI


def main():
    print("=== HVAC Monitor Agent ===")
    mem = AgentMemory("hvac-monitor-01")

    mem.store_event("FAULT_DETECTED",
                    {"room": "301", "unit": "AHU-3", "severity": "high"},
                    entities=[("Room", "Room 301"), ("Equipment", "AHU-3")])
    mem.store_event("WORK_ORDER_CREATED",
                    {"work_order_id": "WO-9921", "assigned_to": "Team A"},
                    entities=[("Equipment", "AHU-3")])
    mem.store_event("FAULT_RESOLVED",
                    {"room": "301", "resolution": "capacitor replaced"},
                    entities=[("Room", "Room 301"), ("Equipment", "AHU-3")])

    print("\n--- Recent events (newest first) ---")
    for e in mem.recall_recent():
        print(f"  [{e['type']}] {e['data']}")

    print("\n--- Events involving Room 301 ---")
    for e in mem.recall_related("Room", "Room 301"):
        print(f"  [{e['type']}] {e['data']}")

    # A second agent writes to the SAME graph — shared memory = coordination.
    print("\n=== Sensor Agent (writes to the same graph) ===")
    sensor = AgentMemory("sensor-01")
    sensor.store_event("TEMPERATURE_ALERT",
                       {"room": "301", "temp": 29.5, "threshold": 26.0},
                       entities=[("Room", "Room 301")])

    print("\n--- All agents now in the graph ---")
    for row in mem.cypher("MATCH (a:Agent) RETURN a.id AS id"):
        print(f"  {row['id']}")

    print("\n--- A Cypher query: count events per room ---")
    for row in mem.cypher(
        "MATCH (e:Event)-[:INVOLVES]->(r:Room) "
        "RETURN r.name AS room, count(e) AS events ORDER BY events DESC"
    ):
        print(f"  {row['room']}: {row['events']} events")

    mem.close()
    sensor.close()
    print("\nDone. Rerun this script — the memory persists. Check http://localhost:7474")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print("⚠️  Could not talk to Neo4j.")
        print(f"   ({type(e).__name__}: {e})")
        print(f"   Expected Neo4j at {URI}. Start it with:")
        print("   docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
