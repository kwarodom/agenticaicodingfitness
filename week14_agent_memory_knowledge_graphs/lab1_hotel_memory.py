"""lab1_hotel_memory.py — Neo4j knowledge-graph memory for the Hotel MAS.

The KG stores the history of every sensor reading and HVAC action, each linked
to the Room (and Device) it concerns. Because the history is a graph, an agent
can later ask relational questions like:

    "Which rooms had more than 3 temperature overrides this month?"
    "Which rooms get the most HVAC actions?"  (predictive-maintenance signal)

Run:
    ./.venv/bin/python week14/lab1_hotel_memory.py        # from repo root

Prerequisites:
    - Neo4j running at bolt://localhost:7687 (docker compose up -d)
    - APOC plugin installed (used to parse JSON inside Cypher)
"""
import os
import sys

# Make agent_memory importable whether run from repo root or from week14/.
sys.path.insert(0, os.path.dirname(__file__))
from agent_memory import AgentMemory

# A single named agent owns this memory stream.
mem = AgentMemory("HotelEnergyMAS")


# ── Store: write events into the knowledge graph ────────────────────────────────
def store_reading(room_id: str, reading: dict):
    """Store a sensor reading as an Event, linked to its Room and HVAC Device."""
    mem.store_event(
        event_type="sensor_reading",
        data=reading,
        entities=[
            ("Room", room_id),
            ("Device", f"HVAC_{room_id[1:]}"),
        ],
    )


def store_action(room_id: str, action: str, setpoint: float):
    """Store an HVAC control action as an Event, linked to its Room."""
    mem.store_event(
        event_type="hvac_action",
        data={"action": action, "setpoint": setpoint},
        entities=[("Room", room_id)],
    )


# ── Recall: ask relational questions of the memory ──────────────────────────────
def rooms_with_recent_alerts(hours: int = 24):
    """Rooms whose recent sensor readings exceeded 27 C, with how many times."""
    return mem.cypher(
        """
        MATCH (e:Event {type: "sensor_reading"})-[:INVOLVES]->(r:Room)
        WHERE e.ts > datetime() - duration({hours: $h})
          AND apoc.convert.fromJsonMap(e.data).temp_c > 27
        RETURN r.name AS room, count(e) AS alert_count
        ORDER BY alert_count DESC
        """,
        h=hours,
    )


def most_active_hvac_rooms(limit: int = 5):
    """Rooms ranked by how many HVAC actions they have received (maintenance signal)."""
    return mem.cypher(
        """
        MATCH (e:Event {type: "hvac_action"})-[:INVOLVES]->(r:Room)
        RETURN r.name AS room, count(e) AS actions
        ORDER BY actions DESC
        LIMIT $lim
        """,
        lim=limit,
    )


# ── Run ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Hotel MAS Memory — store readings/actions, then recall")
    print("=" * 60)

    # Simulate a stream of readings and actions across several rooms.
    # R301 is a repeat offender: 3 hot readings + 2 HVAC actions.
    store_reading("R301", {"temp_c": 28.3, "occupancy": True,  "kwh": 4.1})
    store_action("R301", "HVAC_REDUCED", 22.0)
    store_reading("R301", {"temp_c": 27.6, "occupancy": True,  "kwh": 3.8})
    store_action("R301", "HVAC_REDUCED", 21.5)
    store_reading("R301", {"temp_c": 29.0, "occupancy": True,  "kwh": 4.4})

    store_reading("R412", {"temp_c": 27.9, "occupancy": True,  "kwh": 3.6})
    store_action("R412", "HVAC_REDUCED", 22.5)

    store_reading("R105", {"temp_c": 24.5, "occupancy": False, "kwh": 1.2})  # not an alert

    print("\n--- Rooms with temperature alerts (>27 C, last 24h) ---")
    for row in rooms_with_recent_alerts():
        print(f"  {row['room']}: {row['alert_count']} hot reading(s)")

    print("\n--- Most active HVAC rooms (maintenance signal) ---")
    for row in most_active_hvac_rooms():
        print(f"  {row['room']}: {row['actions']} action(s)")

    print("\n--- Full event history for Room R301 ---")
    for e in mem.recall_related("Room", "R301"):
        print(f"  [{e['type']}] {e['data']}")

    mem.close()
    print("\nDone.")
