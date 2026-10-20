#!/usr/bin/env python3
"""Part 5.4 — Financial fraud detection, Cypher reference (kg_mastery.pdf §5.4).

Fraud hides in the SHAPE of the data: money that loops back to its origin,
ownership webs that obscure a controlling party. Relational SQL struggles with
these variable-length, cyclic patterns; a graph finds them in one query.

╔══════════════════════════════════════════════════════════════════════════╗
║  This file uses ONLY SYNTHETIC, made-up accounts/companies/people.         ║
║  No real financial data. Real AML work needs governance + audit controls.  ║
╚══════════════════════════════════════════════════════════════════════════╝

To avoid touching the hotel data, every node carries an extra `:FIN` label.
Cleanup:  MATCH (n:FIN) DETACH DELETE n;     # --clean

Synthetic subgraph:
  (:FIN:Account {id,owner})-[:TRANSFERS_TO {amount,ts}]->(:FIN:Account)
  (:FIN:Person {id,name})-[:OWNS]->(:FIN:Account)
  (:FIN:Account)-[:CONTROLLED_BY]->(:FIN:Company {id,name})
  (:FIN:Company)-[:AFFILIATED_WITH]->(:FIN:Company)

We build a 4-account TRANSFERS_TO ring (A→B→C→D→A) with large amounts so the
fraud-ring query returns a circular flow.

Queries demonstrated:
  1. Fraud-ring detection  — circular TRANSFERS_TO paths over a min amount.
  2. Counterparty risk     — entities reachable via OWNS|CONTROLLED_BY|AFFILIATED_WITH.

Run:  python week15/kg_mastery/part5_use_cases/04_financial_fraud.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import get_driver, check_connection


def load_synthetic(session):
    """Idempotent loader (MERGE) for the :FIN SYNTHETIC subgraph + a money ring."""
    session.run(
        """
        // ── Accounts ───────────────────────────────────────────────
        MERGE (a:FIN:Account {id: 'ACC_A'}) SET a.owner='Alice'
        MERGE (b:FIN:Account {id: 'ACC_B'}) SET b.owner='Bob'
        MERGE (c:FIN:Account {id: 'ACC_C'}) SET c.owner='Carol'
        MERGE (d:FIN:Account {id: 'ACC_D'}) SET d.owner='Dan'
        MERGE (e:FIN:Account {id: 'ACC_E'}) SET e.owner='Eve'

        // ── Circular money flow A→B→C→D→A (the fraud ring) ─────────
        MERGE (a)-[t1:TRANSFERS_TO]->(b) SET t1.amount=95000, t1.ts=datetime('2026-05-01T09:00:00')
        MERGE (b)-[t2:TRANSFERS_TO]->(c) SET t2.amount=93000, t2.ts=datetime('2026-05-02T10:00:00')
        MERGE (c)-[t3:TRANSFERS_TO]->(d) SET t3.amount=91000, t3.ts=datetime('2026-05-03T11:00:00')
        MERGE (d)-[t4:TRANSFERS_TO]->(a) SET t4.amount=90000, t4.ts=datetime('2026-05-04T12:00:00')

        // ── A normal (non-circular) transfer for contrast ──────────
        MERGE (e)-[t5:TRANSFERS_TO]->(a) SET t5.amount=1200, t5.ts=datetime('2026-05-05T13:00:00')

        // ── Ownership / control / affiliation web ──────────────────
        MERGE (pa:FIN:Person {id: 'P_ALICE'}) SET pa.name='Alice'
        MERGE (pe:FIN:Person {id: 'P_EVE'})   SET pe.name='Eve'
        MERGE (co1:FIN:Company {id: 'CO1'}) SET co1.name='Shell Holdings'
        MERGE (co2:FIN:Company {id: 'CO2'}) SET co2.name='Nominee Ltd'

        MERGE (pa)-[:OWNS]->(a)
        MERGE (pe)-[:OWNS]->(e)
        MERGE (a)-[:CONTROLLED_BY]->(co1)
        MERGE (b)-[:CONTROLLED_BY]->(co1)
        MERGE (co1)-[:AFFILIATED_WITH]->(co2)
        MERGE (e)-[:CONTROLLED_BY]->(co2)
        """
    )
    print("✅ Loaded SYNTHETIC :FIN subgraph with a 4-account money ring (idempotent).")


def fraud_ring_detection(session, min_amount=50000):
    """Find circular TRANSFERS_TO flows (2..5 hops) where every leg is large."""
    rows = [
        r.data()
        for r in session.run(
            """
            MATCH path = (a:FIN:Account)-[:TRANSFERS_TO*2..5]->(a)
            WHERE all(r IN relationships(path) WHERE r.amount >= $min_amount)
            WITH a, path,
                 [n IN nodes(path) | n.id] AS ring,
                 reduce(s = 0, r IN relationships(path) | s + r.amount) AS total_flow,
                 length(path) AS hops
            RETURN DISTINCT a.id AS origin, hops, ring, total_flow
            ORDER BY total_flow DESC
            LIMIT 10
            """,
            min_amount=min_amount,
        )
    ]
    print(f"\n[1] Fraud-ring detection (circular transfers, each leg ≥ {min_amount:,}):")
    if not rows:
        print("    none")
    for r in rows:
        print(f"    {r['hops']}-hop ring {r['ring']}  total≈{r['total_flow']:,}")


def counterparty_risk(session, account_id="ACC_A"):
    """Entities connected to an account via ownership/control/affiliation paths."""
    rows = [
        r.data()
        for r in session.run(
            """
            MATCH (start:FIN:Account {id: $aid})
            MATCH (start)-[:OWNS|CONTROLLED_BY|AFFILIATED_WITH*1..4]-(connected)
            WHERE connected <> start
            RETURN DISTINCT labels(connected)[-1] AS kind,
                   coalesce(connected.name, connected.id) AS entity
            ORDER BY kind, entity
            """,
            aid=account_id,
        )
    ]
    print(f"\n[2] Counterparty risk — entities linked to {account_id}:")
    if not rows:
        print("    none")
    for r in rows:
        print(f"    {r['kind']:<10} {r['entity']}")


# ── GDS Louvain for fraud-ring / community detection (needs GDS plugin) ──────
# Louvain clusters tightly-connected accounts; suspiciously dense clusters of
# transfers are candidate fraud rings to escalate.
#
#   CALL gds.graph.project('txGraph', 'Account',
#     { TRANSFERS_TO: { properties: 'amount' } });
#
#   CALL gds.louvain.stream('txGraph')
#   YIELD nodeId, communityId
#   RETURN communityId,
#          collect(gds.util.asNode(nodeId).id) AS accounts,
#          count(*) AS size
#   ORDER BY size DESC;
#
#   CALL gds.graph.drop('txGraph');


def main():
    driver = get_driver()
    try:
        with driver.session() as s:
            load_synthetic(s)
            fraud_ring_detection(s, min_amount=50000)
            counterparty_risk(s, "ACC_A")
            print("\n(to remove this subgraph: MATCH (n:FIN) DETACH DELETE n;)")
    finally:
        driver.close()


if __name__ == "__main__":
    if not check_connection():
        sys.exit(1)
    main()
