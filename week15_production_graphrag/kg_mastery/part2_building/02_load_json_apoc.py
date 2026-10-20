#!/usr/bin/env python3
"""Part 2.1 — Loading JSON with apoc.load.json (kg_mastery.pdf §2.1).

The PDF shows the APOC loader for nested/array JSON:

    CALL apoc.load.json('file:///iot_readings.json') YIELD value
    MERGE (r:Room {id: value.room_id})
    CREATE (s:SensorReading {
        ts:        datetime(value.timestamp),
        temp_c:    value.temp,
        humidity_pct: value.humidity,
        occupancy: value.occupied
    })
    MERGE (r)-[:HAS_READING]->(s)

Like LOAD CSV, `apoc.load.json('file:///...')` needs the file inside Neo4j's
import dir, which our docker volume doesn't expose. So we read the JSON in
Python and stream it with a parameterized `UNWIND $rows AS value ... CREATE`.
APOC must still be installed for the *raw* form above (it is, per docker-compose),
but this Python path needs no server-side file access.

SensorReading is the same label the curated hotel dataset uses, so these
readings hang off the same :Room nodes (X1xx imported rooms, or whatever ids
the JSON references).

Run:  python week15/kg_mastery/part2_building/02_load_json_apoc.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import get_driver, check_connection

JSON_PATH = Path(__file__).resolve().parent / "sample_data" / "iot_readings.json"


def read_readings(json_path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def load_readings(session, rows):
    """Parameterized UNWIND — the portable equivalent of apoc.load.json."""
    result = session.run(
        """
        UNWIND $rows AS value
        MERGE (r:Room {id: value.room_id})
        ON CREATE SET r:Imported
        CREATE (s:SensorReading {
            ts:           datetime(value.timestamp),
            temp_c:       value.temp,
            humidity_pct: value.humidity,
            occupancy:    value.occupied
        })
        MERGE (r)-[:HAS_READING]->(s)
        RETURN count(s) AS created
        """,
        rows=rows,
    )
    return result.single()["created"]


def main():
    if not JSON_PATH.exists():
        print(f"⚠️  Missing JSON: {JSON_PATH}")
        sys.exit(1)

    rows = read_readings(JSON_PATH)
    print(f"Read {len(rows)} readings from {JSON_PATH.name}")

    driver = get_driver()
    try:
        with driver.session() as s:
            created = load_readings(s, rows)
            print(f"✅ Created {created} :SensorReading nodes linked via HAS_READING.")
    finally:
        driver.close()


if __name__ == "__main__":
    if not check_connection():
        sys.exit(1)
    main()
