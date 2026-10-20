"""Shared helpers for the KG Mastery code companion.

Every script imports the driver + constants from here so connection details and
model IDs live in ONE place. Override via env: NEO4J_URI / NEO4J_USER /
NEO4J_PASSWORD.

The hotel schema contract (what the loader builds and every query expects):

    (:Room {id, floor, type, capacity, rate_thb, status, description})
       -[:HAS_DEVICE]->  (:Device {id, type, model, manufacturer, installed_at, status})
       -[:HAS_READING]-> (:SensorReading {ts, temp_c, humidity_pct, energy_kwh, occupancy})
    (:Device)-[:TRIGGERED]->(:Alert {id, type, severity, message, ts, resolved})
    (:Staff {id, name, role, shift})-[:PERFORMED]->(:MaintenanceJob {id, type, started_at, completed_at, status})
    (:MaintenanceJob)-[:RESOLVES]->(:Alert)
    (:MaintenanceJob)-[:FOR_ROOM]->(:Room)
    (:Staff)-[:ASSIGNED_TO]->(:Alert)
    (:Guest {id, name, check_in, check_out})-[:STAYED_IN]->(:Room)
    (:Supplier {id, name, country, category})-[:PROVIDES]->(:Device)
    (:Agent {id})-[:PERFORMED]->(:Event {id, type, details, ts})
"""
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())  # find the repo-root .env from any subfolder

from neo4j import GraphDatabase

# ── Connection (matches docker-compose.yml) ─────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "mas_memory_2024")
AUTH = (NEO4J_USER, NEO4J_PASSWORD)

# ── Model IDs (swap freely) ─────────────────────────────────────────────────
LLM_MODEL = "claude-sonnet-4-6"            # reasoning / Cypher generation
FAST_MODEL = "claude-haiku-4-5-20251001"   # cheap / high-volume
EMBED_MODEL = "text-embedding-3-small"     # OpenAI, 1536 dims (Part 2.6)


def get_driver():
    """Return a Neo4j driver. Caller is responsible for .close()."""
    return GraphDatabase.driver(NEO4J_URI, auth=AUTH)


def run(cypher, **params):
    """Run one Cypher statement and return a list of dict rows.

    Convenience for quick scripts. For many queries, open one driver/session
    yourself instead of calling this repeatedly.
    """
    driver = get_driver()
    try:
        with driver.session() as s:
            return [r.data() for r in s.run(cypher, **params)]
    finally:
        driver.close()


def check_connection():
    """Verify Neo4j is reachable; print a friendly message if not. Returns bool."""
    try:
        driver = get_driver()
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception as e:  # noqa: BLE001
        print("⚠️  Cannot reach Neo4j.")
        print(f"   ({type(e).__name__}: {e})")
        print(f"   Expected at {NEO4J_URI}. Start it with:")
        print("   docker compose -f week15/kg_mastery/docker-compose.yml up -d")
        return False
