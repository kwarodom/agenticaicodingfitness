#!/usr/bin/env python3
"""Part 2.1 — Loading CSV into Neo4j (kg_mastery.pdf §2.1).

The PDF teaches the canonical Cypher loader:

    LOAD CSV WITH HEADERS FROM 'file:///rooms.csv' AS row
    MERGE (r:Room {id: row.room_id})
    SET r.floor    = toInteger(row.floor),
        r.type     = row.room_type,
        r.capacity = toInteger(row.capacity),
        r.rate_thb = toFloat(row.rate_thb)

BUT `file:///` requires the CSV to live inside Neo4j's own `import/` directory.
Our docker-compose mounts data/plugins but NOT an import volume, so `file:///`
won't see this file. The robust, portable approach — and the one we use here —
is to read the CSV in Python and stream the rows to Neo4j with a parameterized
`UNWIND $rows AS row ... MERGE`. Same result, no server-side file access needed.

These rooms get ids like X101 (not the R1xx curated dataset) and an extra
`:Imported` label so you can tell them apart and clean them up easily:

    MATCH (r:Room:Imported) DETACH DELETE r

Run:  python week15/kg_mastery/part2_building/01_load_csv.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import get_driver, check_connection

CSV_PATH = Path(__file__).resolve().parent / "sample_data" / "rooms.csv"


def read_rows(csv_path):
    """Read rooms.csv and do the type conversions in Python.

    In raw LOAD CSV these would be toInteger()/toFloat() in Cypher. Doing them
    here keeps the server-side query trivial and the data clean.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "room_id": row["room_id"].strip(),
                    "floor": int(row["floor"]),
                    "room_type": row["room_type"].strip(),
                    "capacity": int(row["capacity"]),
                    "rate_thb": float(row["rate_thb"]),
                }
            )
    return rows


def load_rooms(session, rows):
    """Parameterized UNWIND loader — the portable equivalent of LOAD CSV."""
    result = session.run(
        """
        UNWIND $rows AS row
        MERGE (r:Room {id: row.room_id})
        SET r:Imported,
            r.floor    = row.floor,
            r.type     = row.room_type,
            r.capacity = row.capacity,
            r.rate_thb = row.rate_thb
        RETURN count(r) AS loaded
        """,
        rows=rows,
    )
    return result.single()["loaded"]


def main():
    if not CSV_PATH.exists():
        print(f"⚠️  Missing CSV: {CSV_PATH}")
        sys.exit(1)

    rows = read_rows(CSV_PATH)
    print(f"Read {len(rows)} rows from {CSV_PATH.name}")

    driver = get_driver()
    try:
        with driver.session() as s:
            loaded = load_rooms(s, rows)
            print(f"✅ Loaded {loaded} :Room:Imported nodes.")
            print("   Clean up later with: MATCH (r:Room:Imported) DETACH DELETE r")
    finally:
        driver.close()


if __name__ == "__main__":
    if not check_connection():
        sys.exit(1)
    main()
