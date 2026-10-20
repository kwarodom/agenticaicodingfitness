#!/usr/bin/env python3
"""Part 5.2 — Supply-chain risk analysis, Cypher reference (kg_mastery.pdf §5.2).

A knowledge graph is a natural fit for supply chains: the value is in the PATHS
(who supplies what, which products depend on which components, who buys them).
This script is a Cypher REFERENCE plus a tiny SYNTHETIC loader. There is no live
LLM agent here — the point is the impact-analysis queries.

To avoid touching the hotel data, every node carries an extra `:SC` label
(SC = Supply Chain). Cleaning up is one statement:

    MATCH (n:SC) DETACH DELETE n;      # --clean

Synthetic subgraph:
  (:SC:Supplier {id,name,country,reliability_score,lead_time_days})
      -[:SUPPLIES]->(:SC:Component {id,name})
  (:SC:Component)-[:REQUIRED_BY]->(:SC:Product {id,name})
  (:SC:Product)-[:ORDERED_BY]->(:SC:Customer {id,name,country})

Queries demonstrated:
  1. Supplier-disruption impact  — components/products/customers exposed if a
     supplier goes down.
  2. Alternative suppliers       — other suppliers for the same components.
  3. Single-supplier dependencies — components with exactly one supplier (risk).

Run:  python week15/kg_mastery/part5_use_cases/02_supply_chain.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import get_driver, check_connection


def load_synthetic(session):
    """Idempotent loader (MERGE) for the :SC supply-chain subgraph."""
    session.run(
        """
        // ── Suppliers ──────────────────────────────────────────────
        MERGE (s1:SC:Supplier {id: 'SC_SUP1'})
          SET s1.name='Acme Components', s1.country='Thailand',
              s1.reliability_score=0.92, s1.lead_time_days=7
        MERGE (s2:SC:Supplier {id: 'SC_SUP2'})
          SET s2.name='Global Parts Co', s2.country='China',
              s2.reliability_score=0.78, s2.lead_time_days=21
        MERGE (s3:SC:Supplier {id: 'SC_SUP3'})
          SET s3.name='EuroSensor GmbH', s3.country='Germany',
              s3.reliability_score=0.95, s3.lead_time_days=14

        // ── Components ─────────────────────────────────────────────
        MERGE (c1:SC:Component {id: 'CMP1'}) SET c1.name='Temperature Sensor'
        MERGE (c2:SC:Component {id: 'CMP2'}) SET c2.name='Control Board'
        MERGE (c3:SC:Component {id: 'CMP3'}) SET c3.name='Power Supply'

        // ── Products ───────────────────────────────────────────────
        MERGE (p1:SC:Product {id: 'PRD1'}) SET p1.name='Smart Thermostat'
        MERGE (p2:SC:Product {id: 'PRD2'}) SET p2.name='HVAC Controller'

        // ── Customers ──────────────────────────────────────────────
        MERGE (cu1:SC:Customer {id: 'CUS1'}) SET cu1.name='Bangkok Grand Hotel', cu1.country='Thailand'
        MERGE (cu2:SC:Customer {id: 'CUS2'}) SET cu2.name='Singapore Suites',     cu2.country='Singapore'

        // ── Supply edges (SUP1 is sole source of CMP1 → single-supplier risk) ─
        MERGE (s1)-[:SUPPLIES]->(c1)
        MERGE (s2)-[:SUPPLIES]->(c2)
        MERGE (s3)-[:SUPPLIES]->(c2)
        MERGE (s2)-[:SUPPLIES]->(c3)

        // ── Bill of materials ──────────────────────────────────────
        MERGE (c1)-[:REQUIRED_BY]->(p1)
        MERGE (c2)-[:REQUIRED_BY]->(p1)
        MERGE (c2)-[:REQUIRED_BY]->(p2)
        MERGE (c3)-[:REQUIRED_BY]->(p2)

        // ── Orders ─────────────────────────────────────────────────
        MERGE (p1)-[:ORDERED_BY]->(cu1)
        MERGE (p2)-[:ORDERED_BY]->(cu1)
        MERGE (p2)-[:ORDERED_BY]->(cu2)
        """
    )
    print("✅ Loaded synthetic :SC supply-chain subgraph (idempotent).")


def supplier_disruption_impact(session, supplier_id="SC_SUP1"):
    """What breaks if this supplier goes down: components, products, customers."""
    rows = [
        r.data()
        for r in session.run(
            """
            MATCH (s:SC:Supplier {id: $sid})-[:SUPPLIES]->(c:SC:Component)
            OPTIONAL MATCH (c)-[:REQUIRED_BY]->(p:SC:Product)
            OPTIONAL MATCH (p)-[:ORDERED_BY]->(cu:SC:Customer)
            RETURN s.name AS supplier,
                   collect(DISTINCT c.name) AS components,
                   collect(DISTINCT p.name) AS products,
                   collect(DISTINCT cu.name) AS customers
            """,
            sid=supplier_id,
        )
    ]
    print(f"\n[1] Disruption impact if supplier {supplier_id} fails:")
    for r in rows:
        print(f"    supplier : {r['supplier']}")
        print(f"    components: {r['components']}")
        print(f"    products  : {r['products']}")
        print(f"    customers : {r['customers']}")


def alternative_suppliers(session, supplier_id="SC_SUP1"):
    """For each component the supplier provides, find OTHER suppliers (fallbacks)."""
    rows = [
        r.data()
        for r in session.run(
            """
            MATCH (s:SC:Supplier {id: $sid})-[:SUPPLIES]->(c:SC:Component)
            OPTIONAL MATCH (alt:SC:Supplier)-[:SUPPLIES]->(c)
            WHERE alt.id <> $sid
            RETURN c.name AS component,
                   collect(alt.name + ' (' + toString(alt.reliability_score) + ')') AS alternatives
            ORDER BY component
            """,
            sid=supplier_id,
        )
    ]
    print(f"\n[2] Alternative suppliers for components from {supplier_id}:")
    for r in rows:
        alts = r["alternatives"] or ["⚠️  NONE — single point of failure"]
        print(f"    {r['component']:<22} → {alts}")


def single_supplier_dependencies(session):
    """Components served by exactly one supplier — top supply-chain risk."""
    rows = [
        r.data()
        for r in session.run(
            """
            MATCH (c:SC:Component)<-[:SUPPLIES]-(s:SC:Supplier)
            WITH c, count(DISTINCT s) AS supplier_count, collect(s.name) AS suppliers
            WHERE supplier_count = 1
            RETURN c.name AS component, suppliers[0] AS sole_supplier
            ORDER BY component
            """
        )
    ]
    print("\n[3] Single-supplier dependencies (risk):")
    if not rows:
        print("    none")
    for r in rows:
        print(f"    {r['component']:<22} → only {r['sole_supplier']}")


# ── GDS Betweenness Centrality (run after enabling the GDS plugin) ───────────
# Betweenness highlights "broker" nodes that many supply paths pass through —
# the components/suppliers whose failure fragments the network the most.
#
#   CALL gds.graph.project('scGraph',
#     ['Supplier','Component','Product','Customer'],
#     ['SUPPLIES','REQUIRED_BY','ORDERED_BY']);
#
#   CALL gds.betweenness.stream('scGraph')
#   YIELD nodeId, score
#   RETURN gds.util.asNode(nodeId).name AS name, score
#   ORDER BY score DESC LIMIT 10;
#
#   CALL gds.graph.drop('scGraph');


def main():
    driver = get_driver()
    try:
        with driver.session() as s:
            load_synthetic(s)
            supplier_disruption_impact(s, "SC_SUP1")
            alternative_suppliers(s, "SC_SUP1")
            single_supplier_dependencies(s)
            print("\n(to remove this subgraph: MATCH (n:SC) DETACH DELETE n;)")
    finally:
        driver.close()


if __name__ == "__main__":
    if not check_connection():
        sys.exit(1)
    main()
