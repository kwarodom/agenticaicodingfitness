#!/usr/bin/env python3
"""Part 1.4 — Cypher Intermediate (10 patterns), kg_mastery.pdf §1.4.

Ten intermediate Cypher patterns against the hotel IoT graph: variable-length
paths, shortestPath, collect()/comprehensions, CASE, FOREACH, CALL{} and
EXISTS{} subqueries, temporal duration math, and pattern comprehension. Any
demo write uses a `:Demo` label and is cleaned up.

Run:  python part1_fundamentals/02_cypher_intermediate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import get_driver, run, check_connection


def hdr(title):
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


def p1_multi_hop(s):
    hdr("1) Multi-hop variable path HAS_DEVICE|TRIGGERED*1..3")
    rows = [r.data() for r in s.run(
        """
        MATCH path = (r:Room {id:'R101'})-[:HAS_DEVICE|TRIGGERED*1..3]->(n)
        RETURN length(path) AS hops, labels(n)[0] AS endLabel,
               coalesce(n.id, '?') AS endId
        ORDER BY hops
        LIMIT 8
        """)]
    for row in rows:
        print(f"   {row['hops']} hop(s) -> {row['endLabel']} {row['endId']}")


def p2_shortest_path(s):
    hdr("2) shortestPath between two rooms (via shared structure)")
    # Rooms connect to each other only through shared Suppliers/Staff, so a
    # path may not exist. We search undirected up to 6 hops and explain if none.
    rows = [r.data() for r in s.run(
        """
        MATCH (a:Room {id:'R101'}), (b:Room {id:'R203'})
        MATCH p = shortestPath((a)-[*..6]-(b))
        RETURN length(p) AS hops,
               [n IN nodes(p) | labels(n)[0]] AS labelChain
        """)]
    if rows:
        print(f"   shortest = {rows[0]['hops']} hops via {rows[0]['labelChain']}")
    else:
        print("   No path within 6 hops — rooms are only indirectly connected")
        print("   (e.g. through shared Suppliers/Staff). This is expected for")
        print("   a star-shaped hotel schema; pick connected anchors if needed.")


def p3_var_length_max_depth(s):
    hdr("3) Variable-length relationships with a max depth")
    rows = [r.data() for r in s.run(
        """
        MATCH (r:Room {id:'R101'})-[:HAS_DEVICE*1..2]->(d)
        RETURN DISTINCT d.id AS reachable
        ORDER BY reachable
        LIMIT 5
        """)]
    print("   reachable within 2 HAS_DEVICE hops:", [r["reachable"] for r in rows])


def p4_collect_comprehension(s):
    hdr("4) collect() + list comprehension (energy per room)")
    rows = [r.data() for r in s.run(
        """
        MATCH (r:Room)-[:HAS_READING]->(s:SensorReading)
        WITH r, collect(s.energy_kwh) AS kwhs
        RETURN r.id AS id,
               size(kwhs) AS n,
               [x IN kwhs WHERE x > 4.0] AS highReadings
        ORDER BY id
        LIMIT 4
        """)]
    for row in rows:
        print(f"   {row['id']}  readings={row['n']}  >4kWh={row['highReadings']}")


def p5_case(s):
    hdr("5) CASE expression (classify room temp HOT/COLD/COMFORTABLE)")
    rows = [r.data() for r in s.run(
        """
        MATCH (r:Room)-[:HAS_READING]->(s:SensorReading)
        WITH r, max(s.temp_c) AS latest
        RETURN r.id AS id, latest,
               CASE
                 WHEN latest > 28 THEN 'HOT'
                 WHEN latest < 22 THEN 'COLD'
                 ELSE 'COMFORTABLE'
               END AS comfort
        ORDER BY latest DESC
        LIMIT 5
        """)]
    for row in rows:
        print(f"   {row['id']}  max_temp={row['latest']}  -> {row['comfort']}")


def p6_foreach(s):
    hdr("6) FOREACH side-effects on a path (mark :Demo nodes visited)")
    # Build a small demo chain, then FOREACH over its nodes to set a flag.
    s.run(
        """
        CREATE (a:Demo {id:'F1'})-[:HAS_DEVICE]->(b:Demo {id:'F2'})
        CREATE (b)-[:HAS_DEVICE]->(c:Demo {id:'F3'})
        """
    )
    s.run(
        """
        MATCH p = (a:Demo {id:'F1'})-[:HAS_DEVICE*]->(:Demo {id:'F3'})
        FOREACH (n IN nodes(p) | SET n.visited = true)
        """
    )
    rows = [r.data() for r in s.run(
        "MATCH (n:Demo) WHERE n.id STARTS WITH 'F' RETURN n.id AS id, n.visited AS v ORDER BY id")]
    print("   visited flags:", rows)
    s.run("MATCH (n:Demo) WHERE n.id STARTS WITH 'F' DETACH DELETE n")


def p7_call_subquery(s):
    hdr("7) CALL {} correlated subquery (per-room device count)")
    rows = [r.data() for r in s.run(
        """
        MATCH (r:Room)
        CALL {
            WITH r
            MATCH (r)-[:HAS_DEVICE]->(d:Device)
            RETURN count(d) AS devices
        }
        RETURN r.id AS id, devices
        ORDER BY devices DESC, id
        LIMIT 5
        """)]
    for row in rows:
        print(f"   {row['id']}  devices={row['devices']}")


def p8_exists_subquery(s):
    hdr("8) EXISTS {} subquery (rooms with an unresolved alert)")
    rows = [r.data() for r in s.run(
        """
        MATCH (r:Room)
        WHERE EXISTS {
            MATCH (r)-[:HAS_DEVICE]->(:Device)-[:TRIGGERED]->(a:Alert)
            WHERE a.resolved = false
        }
        RETURN r.id AS id
        ORDER BY id
        """)]
    print("   rooms with an open alert:", [r["id"] for r in rows])


def p9_datetime(s):
    hdr("9) date/time functions (alerts in last 7 days)")
    rows = [r.data() for r in s.run(
        """
        MATCH (a:Alert)
        WHERE a.ts >= datetime() - duration({days:7})
        RETURN a.id AS id, a.severity AS sev,
               duration.inDays(a.ts, datetime()).days AS daysAgo
        ORDER BY daysAgo
        """)]
    for row in rows:
        print(f"   {row['id']}  {row['sev']}  {row['daysAgo']}d ago")


def p10_pattern_comprehension(s):
    hdr("10) Pattern comprehension (device ids per room inline)")
    rows = [r.data() for r in s.run(
        """
        MATCH (r:Room)
        RETURN r.id AS id,
               [(r)-[:HAS_DEVICE]->(d:Device) | d.id] AS deviceIds
        ORDER BY id
        LIMIT 4
        """)]
    for row in rows:
        print(f"   {row['id']}  devices={row['deviceIds']}")


def main():
    driver = get_driver()
    try:
        with driver.session() as s:
            p1_multi_hop(s)
            p2_shortest_path(s)
            p3_var_length_max_depth(s)
            p4_collect_comprehension(s)
            p5_case(s)
            p6_foreach(s)
            p7_call_subquery(s)
            p8_exists_subquery(s)
            p9_datetime(s)
            p10_pattern_comprehension(s)
            s.run("MATCH (n:Demo) DETACH DELETE n")  # safety net
            print("\n✅ Done (demo nodes cleaned up).")
    finally:
        driver.close()


if __name__ == "__main__":
    if not check_connection():
        sys.exit(1)
    main()
