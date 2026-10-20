#!/usr/bin/env python3
"""Part 1.6 — Exercises (beginner → advanced), kg_mastery.pdf §1.6.

Six practice exercises against the hotel IoT graph. Each function prints:
  • the question
  • the reference Cypher
  • the live result from the loaded dataset

Run:  python part1_fundamentals/07_exercises.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import get_driver, run, check_connection


def show(n, question, cypher, rows):
    print(f"\n── Exercise {n}: {question} " + "─" * 8)
    print("   Cypher:")
    for line in cypher.strip("\n").splitlines():
        print("     " + line)
    print("   Result:")
    if not rows:
        print("     (no rows)")
    for row in rows:
        print("     " + ", ".join(f"{k}={v}" for k, v in row.items()))


def ex1_rooms_by_floor(s):
    cypher = """
MATCH (r:Room)
RETURN r.id AS id, r.floor AS floor, r.type AS type
ORDER BY r.floor, r.id
LIMIT 8
"""
    rows = [r.data() for r in s.run(cypher)]
    show(1, "All rooms ordered by floor", cypher, rows)


def ex2_hot_readings(s):
    cypher = """
MATCH (r:Room)-[:HAS_READING]->(s:SensorReading)
WHERE s.temp_c > 30
RETURN r.id AS room, round(s.temp_c, 1) AS temp_c
ORDER BY temp_c DESC
LIMIT 8
"""
    rows = [r.data() for r in s.run(cypher)]
    show(2, "Sensor readings with temp_c > 30", cypher, rows)


def ex3_device_count(s):
    cypher = """
MATCH (r:Room)-[:HAS_DEVICE]->(d:Device)
RETURN r.id AS room, count(d) AS devices
ORDER BY devices DESC, room
LIMIT 8
"""
    rows = [r.data() for r in s.run(cypher)]
    show(3, "Device count per room (desc)", cypher, rows)


def ex4_rooms_with_recent_alerts(s):
    # Threshold adapted to the dataset: rooms with >= 1 alert in the last 7 days
    cypher = """
MATCH (r:Room)-[:HAS_DEVICE]->(:Device)-[:TRIGGERED]->(a:Alert)
WHERE a.ts >= datetime() - duration({days:7})
WITH r, count(a) AS alerts
WHERE alerts >= 1
RETURN r.id AS room, alerts
ORDER BY alerts DESC, room
"""
    rows = [r.data() for r in s.run(cypher)]
    show(4, "Rooms with >=1 alert in last 7 days (threshold adapted)", cypher, rows)


def ex5_room_to_job_paths(s):
    cypher = """
MATCH p = (r:Room)-[*1..3]-(j:MaintenanceJob)
WHERE r.id = 'R305'
RETURN r.id AS room, j.id AS job, length(p) AS hops
ORDER BY hops
LIMIT 8
"""
    rows = [r.data() for r in s.run(cypher)]
    show(5, "Paths between a Room and a MaintenanceJob within 3 hops", cypher, rows)


def ex6_avg_energy_per_floor(s):
    cypher = """
MATCH (r:Room)-[:HAS_READING]->(s:SensorReading)
WHERE s.ts >= datetime() - duration({hours:24})
RETURN r.floor AS floor, round(avg(s.energy_kwh), 2) AS avg_kwh
ORDER BY avg_kwh DESC
"""
    rows = [r.data() for r in s.run(cypher)]
    show(6, "Per-floor avg energy_kwh in last 24h (desc)", cypher, rows)


def main():
    driver = get_driver()
    try:
        with driver.session() as s:
            ex1_rooms_by_floor(s)
            ex2_hot_readings(s)
            ex3_device_count(s)
            ex4_rooms_with_recent_alerts(s)
            ex5_room_to_job_paths(s)
            ex6_avg_energy_per_floor(s)
            print("\n✅ Exercises complete.")
    finally:
        driver.close()


if __name__ == "__main__":
    if not check_connection():
        sys.exit(1)
    main()
