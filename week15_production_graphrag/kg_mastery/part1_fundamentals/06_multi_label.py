#!/usr/bin/env python3
"""Part 1.9 — Multi-Label Nodes, kg_mastery.pdf §1.9.

A node can carry several labels at once. This lets you query at different levels
of specificity (broad vs. narrow) and combine labels for intersections. All
work here is on temporary `:Demo` nodes that are removed at the end.

Label best practices:
  • Use a broad label for the entity type (:Device) and add narrower labels for
    sub-types (:HVACUnit) and cross-cutting concerns (:SmartDevice).
  • Keep label sets small and stable; prefer properties for high-cardinality or
    frequently-changing attributes (don't make a label per serial number).
  • Use a dynamic/state label (e.g. :MaintenanceRequired) you add and remove to
    flag workflow state cheaply — but back it with constraints/indexes if hot.
  • PascalCase, singular nouns. Index the labels you filter on.

Run:  python part1_fundamentals/06_multi_label.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import get_driver, run, check_connection


def hdr(title):
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


def create_multilabel(s):
    hdr("Create a node with 4 labels :Demo:Device:HVACUnit:SmartDevice")
    s.run(
        """
        CREATE (d:Demo:Device:HVACUnit:SmartDevice {
            id:'DEMO-HVAC-1', model:'Daikin VRV-IV', manufacturer:'Daikin'
        })
        """
    )
    rows = [r.data() for r in s.run(
        "MATCH (d:Demo {id:'DEMO-HVAC-1'}) RETURN labels(d) AS labels")]
    print("   labels:", rows[0]["labels"])


def query_levels(s):
    hdr("Query at broad, specific, and intersection levels")
    broad = [r["id"] for r in s.run(
        "MATCH (d:Demo:Device) RETURN d.id AS id")]
    specific = [r["id"] for r in s.run(
        "MATCH (d:Demo:HVACUnit) RETURN d.id AS id")]
    intersection = [r["id"] for r in s.run(
        "MATCH (d:Demo:SmartDevice:HVACUnit) RETURN d.id AS id")]
    print("   broad      (:Device)            :", broad)
    print("   specific   (:HVACUnit)          :", specific)
    print("   intersect  (:SmartDevice:HVACUnit):", intersection)


def dynamic_label(s):
    hdr("Add a label dynamically, then remove it")
    s.run("MATCH (d:Demo {id:'DEMO-HVAC-1'}) SET d:MaintenanceRequired")
    after_add = [r.data() for r in s.run(
        "MATCH (d:Demo:MaintenanceRequired {id:'DEMO-HVAC-1'}) RETURN labels(d) AS labels")]
    print("   after SET d:MaintenanceRequired ->", after_add[0]["labels"])

    s.run("MATCH (d:Demo {id:'DEMO-HVAC-1'}) REMOVE d:MaintenanceRequired")
    still_flagged = [r.data() for r in s.run(
        "MATCH (d:Demo:MaintenanceRequired {id:'DEMO-HVAC-1'}) RETURN d.id AS id")]
    print("   after REMOVE -> still flagged?", bool(still_flagged))


def cleanup(s):
    s.run("MATCH (n:Demo) DETACH DELETE n")
    print("\n✅ Cleaned up all :Demo nodes.")


def main():
    driver = get_driver()
    try:
        with driver.session() as s:
            create_multilabel(s)
            query_levels(s)
            dynamic_label(s)
            cleanup(s)
    finally:
        driver.close()


if __name__ == "__main__":
    if not check_connection():
        sys.exit(1)
    main()
