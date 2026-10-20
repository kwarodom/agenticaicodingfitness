#!/usr/bin/env python3
"""CHECKPOINT 1 (0:30–0:45) — Neo4j Schema + Seed Data.

Goal: a running Neo4j with 200 rooms, 400 devices, and 2000 sensor readings
(30 days of history). Verify with 3 count queries. (smart_hotel_mas.pdf §CP1)

Run:  python week15/smart_hotel_mas/checkpoints/checkpoint1_seed.py

Schema: constraints on Room/Device/Staff/Guest/Agent ids; indexes on
SensorReading.ts and Alert(ts,severity). Rooms across 5 floors (40/floor),
each with an HVAC + a SENSOR device; SENSOR devices RECORDED 10 readings each.

This is idempotent: rooms/devices use MERGE, and SensorReadings are cleared
before re-seeding so re-runs always yield exactly 2000.
"""
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import config.py
from config import get_driver, check_neo4j

random.seed(42)  # reproducible seed data


def setup_schema(s):
    s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Room) REQUIRE r.id IS UNIQUE")
    s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Device) REQUIRE d.id IS UNIQUE")
    s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (st:Staff) REQUIRE st.id IS UNIQUE")
    s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Guest) REQUIRE g.id IS UNIQUE")
    s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Agent) REQUIRE a.id IS UNIQUE")
    s.run("CREATE INDEX IF NOT EXISTS FOR (sr:SensorReading) ON (sr.ts)")
    s.run("CREATE INDEX IF NOT EXISTS FOR (al:Alert) ON (al.ts, al.severity)")
    s.run("CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.type, e.ts)")
    print("Schema constraints and indexes created.")


def seed_rooms(s):
    for floor in range(1, 6):            # 5 floors
        for room_num in range(1, 41):    # 40 rooms per floor → 200 rooms
            room_id = f"R{floor}{room_num:02d}"
            room_type = random.choice(["standard", "deluxe", "suite"])
            s.run(
                """
                MERGE (r:Room {id: $rid})
                SET r.floor = $floor, r.type = $rtype,
                    r.capacity = $cap, r.rate_thb = $rate, r.status = 'active'
                """,
                rid=room_id, floor=floor, rtype=room_type,
                cap=2 if room_type == "suite" else 1,
                rate=random.randint(2000, 8000),
            )
            # each room gets an HVAC + a SENSOR device
            for dtype in ["HVAC", "SENSOR"]:
                dev_id = f"{dtype}_{room_id}"
                s.run(
                    """
                    MERGE (d:Device {id: $did})
                    SET d.type = $dtype, d.room_id = $rid, d.status = 'active'
                    WITH d
                    MATCH (r:Room {id: $rid})
                    MERGE (r)-[:HAS_DEVICE]->(d)
                    """,
                    did=dev_id, dtype=dtype, rid=room_id,
                )
    print("Seeded 200 rooms + 400 devices.")


def seed_readings(s):
    # clear old readings so re-runs are deterministic (CREATE would duplicate)
    s.run("MATCH (sr:SensorReading) DETACH DELETE sr")
    base_date = datetime.now() - timedelta(days=30)
    for floor in range(1, 6):
        for room_num in range(1, 41):
            room_id = f"R{floor}{room_num:02d}"
            for _ in range(10):          # 10 readings per room → 2000 total
                ts = base_date + timedelta(
                    days=random.randint(0, 29), hours=random.randint(0, 23)
                )
                temp = round(20 + random.gauss(4, 2), 1)
                s.run(
                    """
                    MATCH (d:Device {id: $did, type: 'SENSOR'})
                    CREATE (sr:SensorReading {
                        ts: datetime($ts), temp_c: $temp, humidity: $hum,
                        occupancy: $occ, kwh: $kwh
                    })
                    CREATE (d)-[:RECORDED]->(sr)
                    """,
                    did=f"SENSOR_{room_id}", ts=ts.isoformat(), temp=temp,
                    hum=round(40 + random.random() * 30, 1),
                    occ=random.choice([True, False]),
                    kwh=round(1 + random.random() * 3, 2),
                )
    print("Seeded 2000 sensor readings.")


def verify(s):
    print("\n── Verification (expected: 200 / 400 / 2000) ──")
    for label, expected in [("Room", 200), ("Device", 400), ("SensorReading", 2000)]:
        got = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
        flag = "✅" if got == expected else "⚠️"
        print(f"  {flag} {label}: {got}")


def seed_database():
    driver = get_driver()
    try:
        with driver.session() as s:
            setup_schema(s)
            seed_rooms(s)
            seed_readings(s)
            verify(s)
        print("\n✅ Checkpoint 1 complete. Explore at http://localhost:7474")
        print("   Next: python checkpoints/checkpoint2_memory.py")
    finally:
        driver.close()


if __name__ == "__main__":
    if not check_neo4j():
        sys.exit(1)
    seed_database()
