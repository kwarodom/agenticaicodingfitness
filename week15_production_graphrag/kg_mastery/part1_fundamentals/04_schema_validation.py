#!/usr/bin/env python3
"""Part 1.7 — Cypher Schema Validation Queries, kg_mastery.pdf §1.7.

A set of audit checks you run against a knowledge graph to catch data-quality
problems early. Each check prints PASS or a ⚠️ warning with the offending rows.

Checks:
  1. Orphan nodes (no relationships, excluding :Chunk)
  2. Nodes missing required properties (Room without id or floor)
  3. Duplicate rooms by id
  4. Degree distribution for Device

Run:  python part1_fundamentals/04_schema_validation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import get_driver, run, check_connection


def hdr(title):
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


def check_orphans(s):
    hdr("AUDIT 1: orphan nodes (no relationships, excluding :Chunk)")
    rows = [r.data() for r in s.run(
        """
        MATCH (n)
        WHERE NOT (n)--() AND NOT n:Chunk
        RETURN labels(n) AS labels, count(*) AS n
        """)]
    if not rows or all(r["n"] == 0 for r in rows):
        print("   PASS ✅  no orphan nodes")
    else:
        print("   ⚠️  orphans found:")
        for r in rows:
            print(f"     {r['labels']}: {r['n']}")


def check_missing_props(s):
    hdr("AUDIT 2: Rooms missing required props (id or floor)")
    rows = [r.data() for r in s.run(
        """
        MATCH (r:Room)
        WHERE r.id IS NULL OR r.floor IS NULL
        RETURN elementId(r) AS eid, r.id AS id, r.floor AS floor
        """)]
    if not rows:
        print("   PASS ✅  every Room has id and floor")
    else:
        print(f"   ⚠️  {len(rows)} Room(s) missing required props:")
        for r in rows:
            print(f"     eid={r['eid']}  id={r['id']}  floor={r['floor']}")


def check_duplicate_rooms(s):
    hdr("AUDIT 3: duplicate rooms by id")
    rows = [r.data() for r in s.run(
        """
        MATCH (r:Room)
        WITH r.id AS id, count(*) AS c
        WHERE c > 1
        RETURN id, c
        ORDER BY c DESC
        """)]
    if not rows:
        print("   PASS ✅  no duplicate Room ids")
    else:
        print(f"   ⚠️  {len(rows)} duplicated id(s):")
        for r in rows:
            print(f"     id={r['id']}  copies={r['c']}")


def check_device_degree(s):
    hdr("AUDIT 4: degree distribution for Device")
    rows = [r.data() for r in s.run(
        """
        MATCH (d:Device)
        WITH d, COUNT { (d)--() } AS degree
        RETURN degree, count(*) AS devices
        ORDER BY degree
        """)]
    if not rows:
        print("   ⚠️  no Device nodes found — did the loader run?")
        return
    print("   degree : devices")
    for r in rows:
        print(f"     {r['degree']:>5}  : {r['devices']}")
    isolated = [r for r in rows if r["degree"] == 0]
    if isolated:
        print(f"   ⚠️  {isolated[0]['devices']} Device(s) are isolated (degree 0)")
    else:
        print("   PASS ✅  every Device is connected")


def main():
    driver = get_driver()
    try:
        with driver.session() as s:
            check_orphans(s)
            check_missing_props(s)
            check_duplicate_rooms(s)
            check_device_degree(s)
            print("\n✅ Schema audit complete.")
    finally:
        driver.close()


if __name__ == "__main__":
    if not check_connection():
        sys.exit(1)
    main()
