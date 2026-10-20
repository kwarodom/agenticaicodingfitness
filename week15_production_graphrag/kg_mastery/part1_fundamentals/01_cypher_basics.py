#!/usr/bin/env python3
"""Part 1.3 — Cypher Basics (12 patterns), kg_mastery.pdf §1.3.

Twelve foundational Cypher patterns, every one adapted to the hotel IoT graph
and RUNNABLE against the loaded dataset. Anything we CREATE/MERGE for a demo is
tagged with a temporary `:Demo` label and DETACH DELETE'd at the end so the
hotel dataset is never polluted.

Run:  python part1_fundamentals/01_cypher_basics.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import get_driver, run, check_connection


def hdr(title):
    """Print a short labelled section header."""
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


def pattern_01_create_node(s):
    hdr("1) CREATE a node (then clean up)")
    s.run("CREATE (d:Demo:Room {id:'DEMO-X', floor:99, type:'Demo'})")
    rows = [r.data() for r in s.run(
        "MATCH (d:Demo {id:'DEMO-X'}) RETURN d.id AS id, d.floor AS floor")]
    print("   created:", rows)
    s.run("MATCH (d:Demo {id:'DEMO-X'}) DETACH DELETE d")


def pattern_02_create_relationship(s):
    hdr("2) CREATE a relationship between two temp nodes")
    s.run(
        """
        CREATE (r:Demo:Room {id:'DEMO-R'})
        CREATE (d:Demo:Device {id:'DEMO-D'})
        CREATE (r)-[:HAS_DEVICE]->(d)
        """
    )
    rows = [r.data() for r in s.run(
        """
        MATCH (r:Demo:Room)-[:HAS_DEVICE]->(d:Demo:Device)
        RETURN r.id AS room, d.id AS device
        """)]
    print("   linked:", rows)
    s.run("MATCH (n:Demo) WHERE n.id IN ['DEMO-R','DEMO-D'] DETACH DELETE n")


def pattern_03_match_order_limit(s):
    hdr("3) MATCH + RETURN with ORDER BY / LIMIT (rooms by floor)")
    rows = [r.data() for r in s.run(
        """
        MATCH (r:Room)
        RETURN r.id AS id, r.floor AS floor, r.type AS type
        ORDER BY r.floor DESC, r.id
        LIMIT 5
        """)]
    for row in rows:
        print(f"   {row['id']}  floor={row['floor']}  {row['type']}")


def pattern_04_where_filter(s):
    hdr("4) WHERE filter (STARTS WITH + numeric on readings)")
    rows = [r.data() for r in s.run(
        """
        MATCH (r:Room)-[:HAS_READING]->(s:SensorReading)
        WHERE r.id STARTS WITH 'R1' AND s.temp_c > 28
        RETURN r.id AS id, max(s.temp_c) AS hottest
        ORDER BY hottest DESC
        """)]
    print("   floor-1 rooms with a HOT reading:", rows or "(none)")


def pattern_05_merge_upsert(s):
    hdr("5) MERGE upsert with ON CREATE / ON MATCH")
    for i in (1, 2):  # second pass should hit ON MATCH
        rows = [r.data() for r in s.run(
            """
            MERGE (d:Demo:Room {id:'DEMO-MERGE'})
            ON CREATE SET d.hits = 1, d.note = 'created'
            ON MATCH  SET d.hits = coalesce(d.hits,0) + 1, d.note = 'matched'
            RETURN d.hits AS hits, d.note AS note
            """)]
        print(f"   pass {i}: {rows[0]}")
    s.run("MATCH (d:Demo {id:'DEMO-MERGE'}) DETACH DELETE d")


def pattern_06_with_aggregation(s):
    hdr("6) WITH + aggregation (devices per room, HAVING-style WHERE)")
    rows = [r.data() for r in s.run(
        """
        MATCH (r:Room)-[:HAS_DEVICE]->(d:Device)
        WITH r, count(d) AS devices
        WHERE devices >= 2
        RETURN r.id AS id, devices
        ORDER BY devices DESC, id
        LIMIT 5
        """)]
    for row in rows:
        print(f"   {row['id']}  devices={row['devices']}")


def pattern_07_unwind_merge(s):
    hdr("7) UNWIND a Python list to MERGE temp nodes")
    amenities = ["wifi", "minibar", "balcony"]
    rows = [r.data() for r in s.run(
        """
        UNWIND $items AS name
        MERGE (a:Demo:Amenity {name:name})
        RETURN count(a) AS created
        """, items=amenities)]
    print("   merged amenities:", rows[0]["created"])
    s.run("MATCH (a:Demo:Amenity) DETACH DELETE a")


def pattern_08_delete(s):
    hdr("8) DELETE / DETACH DELETE")
    s.run("CREATE (a:Demo {id:'DEL-A'})-[:HAS_DEVICE]->(b:Demo {id:'DEL-B'})")
    before = [r.data() for r in s.run("MATCH (n:Demo) RETURN count(n) AS c")][0]["c"]
    # DETACH DELETE removes the node AND its relationships in one step
    s.run("MATCH (n:Demo) WHERE n.id IN ['DEL-A','DEL-B'] DETACH DELETE n")
    after = [r.data() for r in s.run("MATCH (n:Demo) RETURN count(n) AS c")][0]["c"]
    print(f"   demo nodes before={before} after DETACH DELETE={after}")


def pattern_09_set_remove(s):
    hdr("9) SET / REMOVE properties")
    s.run("CREATE (d:Demo:Room {id:'DEMO-SET'})")
    s.run("MATCH (d:Demo {id:'DEMO-SET'}) SET d.status='VACANT', d.temp_c=21.5")
    before = [r.data() for r in s.run(
        "MATCH (d:Demo {id:'DEMO-SET'}) RETURN d.status AS status, d.temp_c AS temp")]
    s.run("MATCH (d:Demo {id:'DEMO-SET'}) REMOVE d.temp_c")
    after = [r.data() for r in s.run(
        "MATCH (d:Demo {id:'DEMO-SET'}) RETURN d.status AS status, d.temp_c AS temp")]
    print(f"   after SET   : {before[0]}")
    print(f"   after REMOVE: {after[0]}")
    s.run("MATCH (d:Demo {id:'DEMO-SET'}) DETACH DELETE d")


def pattern_10_optional_match(s):
    hdr("10) OPTIONAL MATCH (rooms and their maybe-missing alerts)")
    rows = [r.data() for r in s.run(
        """
        MATCH (r:Room)
        OPTIONAL MATCH (r)-[:HAS_DEVICE]->(:Device)-[:TRIGGERED]->(a:Alert)
        RETURN r.id AS id, count(a) AS alerts
        ORDER BY alerts DESC, id
        LIMIT 5
        """)]
    for row in rows:
        print(f"   {row['id']}  alerts={row['alerts']}")


def pattern_11_named_relationship(s):
    hdr("11) Named relationship [r] returning a rel/edge property")
    # TRIGGERED is a plain edge; we read a property off the Alert it points to
    rows = [r.data() for r in s.run(
        """
        MATCH (d:Device)-[t:TRIGGERED]->(a:Alert)
        RETURN d.id AS device, type(t) AS rel, a.severity AS severity
        ORDER BY a.severity
        LIMIT 5
        """)]
    for row in rows:
        print(f"   {row['device']} -[{row['rel']}]-> severity={row['severity']}")


def pattern_12_exists_count(s):
    hdr("12) Counting + EXISTS subquery / size() pattern count")
    rows = [r.data() for r in s.run(
        """
        MATCH (r:Room)
        WHERE EXISTS {
            MATCH (r)-[:HAS_DEVICE]->(:Device)-[:TRIGGERED]->(a:Alert)
            WHERE a.resolved = false
        }
        RETURN r.id AS id,
               COUNT { (r)-[:HAS_DEVICE]->(:Device) } AS devices
        ORDER BY id
        """)]
    print("   rooms with an UNRESOLVED alert:")
    for row in rows:
        print(f"   {row['id']}  devices={row['devices']}")


def main():
    driver = get_driver()
    try:
        with driver.session() as s:
            pattern_01_create_node(s)
            pattern_02_create_relationship(s)
            pattern_03_match_order_limit(s)
            pattern_04_where_filter(s)
            pattern_05_merge_upsert(s)
            pattern_06_with_aggregation(s)
            pattern_07_unwind_merge(s)
            pattern_08_delete(s)
            pattern_09_set_remove(s)
            pattern_10_optional_match(s)
            pattern_11_named_relationship(s)
            pattern_12_exists_count(s)
            # safety net: remove any stray :Demo nodes
            s.run("MATCH (n:Demo) DETACH DELETE n")
            print("\n✅ All demo nodes cleaned up.")
    finally:
        driver.close()


if __name__ == "__main__":
    if not check_connection():
        sys.exit(1)
    main()
