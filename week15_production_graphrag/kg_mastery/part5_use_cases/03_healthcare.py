#!/usr/bin/env python3
"""Part 5.3 — Healthcare knowledge graph, Cypher reference (kg_mastery.pdf §5.3).

Clinical data is deeply relational: patients ↔ diagnoses ↔ medications ↔
treatments, plus drug–drug interactions. A graph lets you ask "does this
patient's medication list contain an interacting pair?" or "which patients look
like this one?" in a single traversal.

╔══════════════════════════════════════════════════════════════════════════╗
║  COMPLIANCE: HIPAA / GDPR / PDPA.                                          ║
║  Real patient data is protected health information. This file uses ONLY    ║
║  fully SYNTHETIC, de-identified, made-up records — no real people, no PHI. ║
║  Never load real clinical data into a demo graph without proper consent,   ║
║  access controls, encryption, and a lawful basis for processing.           ║
╚══════════════════════════════════════════════════════════════════════════╝

To avoid touching the hotel data, every node carries an extra `:HC` label.
Cleanup:  MATCH (n:HC) DETACH DELETE n;      # --clean

Synthetic subgraph:
  (:HC:Patient {id,name,age,sex})-[:HAS_DIAGNOSIS]->(:HC:Diagnosis {id,name})
  (:HC:Patient)-[:PRESCRIBED]->(:HC:Medication {id,name})
  (:HC:Medication)-[:INTERACTS_WITH {severity}]->(:HC:Medication)
  (:HC:Patient)-[:RECEIVED]->(:HC:Treatment {id,name})
  (:HC:Patient)-[:RESPONDED_POSITIVELY]->(:HC:Treatment)

Queries demonstrated:
  1. Drug-interaction detection — patients on an interacting medication pair.
  2. Similar patients (cohort)   — patients sharing diagnoses, for treatment ideas.

Run:  python week15/kg_mastery/part5_use_cases/03_healthcare.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import get_driver, check_connection


def load_synthetic(session):
    """Idempotent loader (MERGE) for the :HC SYNTHETIC, de-identified subgraph."""
    session.run(
        """
        // ── Patients (synthetic) ───────────────────────────────────
        MERGE (p1:HC:Patient {id: 'PT1'}) SET p1.name='Patient A', p1.age=67, p1.sex='F'
        MERGE (p2:HC:Patient {id: 'PT2'}) SET p2.name='Patient B', p2.age=72, p2.sex='M'
        MERGE (p3:HC:Patient {id: 'PT3'}) SET p3.name='Patient C', p3.age=59, p3.sex='F'

        // ── Diagnoses ──────────────────────────────────────────────
        MERGE (d1:HC:Diagnosis {id: 'DX1'}) SET d1.name='Atrial Fibrillation'
        MERGE (d2:HC:Diagnosis {id: 'DX2'}) SET d2.name='Type 2 Diabetes'
        MERGE (d3:HC:Diagnosis {id: 'DX3'}) SET d3.name='Hypertension'

        // ── Medications ────────────────────────────────────────────
        MERGE (m1:HC:Medication {id: 'MED1'}) SET m1.name='Warfarin'
        MERGE (m2:HC:Medication {id: 'MED2'}) SET m2.name='Aspirin'
        MERGE (m3:HC:Medication {id: 'MED3'}) SET m3.name='Metformin'

        // ── Treatments ─────────────────────────────────────────────
        MERGE (t1:HC:Treatment {id: 'TX1'}) SET t1.name='Anticoagulation Therapy'
        MERGE (t2:HC:Treatment {id: 'TX2'}) SET t2.name='Lifestyle + Metformin'

        // ── Known drug–drug interaction (synthetic clinical fact) ──
        MERGE (m1)-[ix:INTERACTS_WITH]->(m2) SET ix.severity='HIGH'

        // ── Patient ↔ diagnosis ────────────────────────────────────
        MERGE (p1)-[:HAS_DIAGNOSIS]->(d1)
        MERGE (p1)-[:HAS_DIAGNOSIS]->(d3)
        MERGE (p2)-[:HAS_DIAGNOSIS]->(d1)
        MERGE (p2)-[:HAS_DIAGNOSIS]->(d2)
        MERGE (p3)-[:HAS_DIAGNOSIS]->(d2)

        // ── Patient ↔ medication (PT1 is on the interacting pair) ──
        MERGE (p1)-[:PRESCRIBED]->(m1)
        MERGE (p1)-[:PRESCRIBED]->(m2)
        MERGE (p2)-[:PRESCRIBED]->(m1)
        MERGE (p3)-[:PRESCRIBED]->(m3)

        // ── Treatments + responses ─────────────────────────────────
        MERGE (p2)-[:RECEIVED]->(t1)
        MERGE (p2)-[:RESPONDED_POSITIVELY]->(t1)
        MERGE (p3)-[:RECEIVED]->(t2)
        MERGE (p3)-[:RESPONDED_POSITIVELY]->(t2)
        """
    )
    print("✅ Loaded SYNTHETIC :HC healthcare subgraph (idempotent, de-identified).")


def drug_interaction_detection(session):
    """Patients prescribed two medications that are known to interact."""
    rows = [
        r.data()
        for r in session.run(
            """
            MATCH (p:HC:Patient)-[:PRESCRIBED]->(m1:HC:Medication)
                  -[ix:INTERACTS_WITH]-(m2:HC:Medication)<-[:PRESCRIBED]-(p)
            WHERE m1.id < m2.id            // dedupe the symmetric pair
            RETURN p.name AS patient,
                   m1.name AS drug_a, m2.name AS drug_b,
                   ix.severity AS severity
            ORDER BY severity DESC, patient
            """
        )
    ]
    print("\n[1] Drug-interaction detection (patients on an interacting pair):")
    if not rows:
        print("    none")
    for r in rows:
        print(f"    {r['patient']:<12} {r['drug_a']} + {r['drug_b']}  [{r['severity']}]")


def similar_patients(session, patient_id="PT1"):
    """Cohort: patients sharing diagnoses with the target, ranked by overlap.

    Surfaces treatments that worked for similar patients — decision support, not
    a prescription.
    """
    rows = [
        r.data()
        for r in session.run(
            """
            MATCH (target:HC:Patient {id: $pid})-[:HAS_DIAGNOSIS]->(d:HC:Diagnosis)
                  <-[:HAS_DIAGNOSIS]-(peer:HC:Patient)
            WHERE peer.id <> $pid
            WITH peer, count(DISTINCT d) AS shared_dx, collect(DISTINCT d.name) AS shared
            OPTIONAL MATCH (peer)-[:RESPONDED_POSITIVELY]->(t:HC:Treatment)
            RETURN peer.name AS similar_patient, shared_dx, shared,
                   collect(DISTINCT t.name) AS treatments_that_worked
            ORDER BY shared_dx DESC
            """,
            pid=patient_id,
        )
    ]
    print(f"\n[2] Similar patients to {patient_id} (shared diagnoses → what worked):")
    if not rows:
        print("    none")
    for r in rows:
        print(
            f"    {r['similar_patient']:<12} shared={r['shared_dx']} {r['shared']} "
            f"→ worked: {r['treatments_that_worked']}"
        )


def main():
    driver = get_driver()
    try:
        with driver.session() as s:
            load_synthetic(s)
            drug_interaction_detection(s)
            similar_patients(s, "PT1")
            print("\n(to remove this subgraph: MATCH (n:HC) DETACH DELETE n;)")
    finally:
        driver.close()


if __name__ == "__main__":
    if not check_connection():
        sys.exit(1)
    main()
