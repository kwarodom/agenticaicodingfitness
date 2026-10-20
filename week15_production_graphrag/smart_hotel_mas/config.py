"""Shared config for the Smart Hotel MAS workshop.

The PDF shows each checkpoint creating its own driver inline. We centralise the
connection settings here so the password/host live in ONE place; every script
imports from this module. Override anything via environment variables.

Hotel KG schema (10 node labels, 15 relationship types):
    (:Room {id, floor, type, capacity, rate_thb, status})
       -[:HAS_DEVICE]->   (:Device {id, type, room_id, model, status})
    (:Device {type:'SENSOR'})-[:RECORDED]->(:SensorReading {ts, temp_c, humidity, occupancy, kwh})
    (:SensorReading)-[:TRIGGERED]->(:Alert {type, severity, threshold, value, ts})
    (:Alert)-[:RESOLVED_BY]->(:MaintenanceJob {type, priority, status, assigned_to, ts})
    (:MaintenanceJob)-[:ASSIGNED_TO]->(:Staff {id, name, role, shift, skills})
    (:Room)-[:OCCUPIED_BY]->(:Guest {id, checkin, checkout, preferences})
    (:Agent {id, role, last_active})-[:PERFORMED]->(:Event {type, details, ts})
    (:Event)-[:INVOLVES]->(:Room|:Device|:Alert|:Guest)
    (:EnergyEvent {action, setpoint, saving_kwh, ts})-[:APPLIED_TO]->(:Room)
    (:Alert)-[:AFFECTS]->(:Room)   (used by CP6 agent tools)
"""
import os

# .env loading is optional — config must import with only the stdlib so that
# dependency-free checkpoints (e.g. CP2: dict + SQLite) run with no installs.
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass

# ── Neo4j (L4) ──────────────────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "hotel_mas_2024")
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)

# ── ChromaDB (L3) ───────────────────────────────────────────────────────────
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))

# ── SQLite (L2) ─────────────────────────────────────────────────────────────
EPISODIC_DB = os.getenv("EPISODIC_DB", "hotel_episodes.db")

# ── LLM ─────────────────────────────────────────────────────────────────────
# PDF predates Opus 4.7 and shows claude-3-5-sonnet-20241022; use the current id.
MODEL = os.getenv("HOTEL_MAS_MODEL", "claude-sonnet-4-6")


def get_driver():
    """Neo4j driver (caller closes it). Imported lazily so config stays stdlib-only."""
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


def check_neo4j():
    """Return True if Neo4j is reachable; print a friendly hint if not."""
    try:
        d = get_driver()
        d.verify_connectivity()
        d.close()
        return True
    except Exception as e:  # noqa: BLE001
        print("⚠️  Cannot reach Neo4j.")
        print(f"   ({type(e).__name__}: {e})")
        print(f"   Expected at {NEO4J_URI} (user {NEO4J_USER}). Start it with:")
        print("   docker compose -f week15/smart_hotel_mas/docker-compose.yml up -d")
        return False
