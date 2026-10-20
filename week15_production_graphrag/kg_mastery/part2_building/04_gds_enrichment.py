#!/usr/bin/env python3
"""Part 2.5 — Graph enrichment with Graph Data Science (kg_mastery.pdf §2.5).

Runs three classic GDS algorithms over a projected subgraph of the hotel graph
and writes the results back as node properties:

  - PageRank     → r/d/a.importance_score   (which Devices/Rooms/Alerts are central)
  - Louvain      → .community_id            (community detection)
  - Node2Vec     → .n2v_embedding (dim 64)  (structural embeddings for ML / similarity)

We project a named graph 'deviceGraph' over the labels ['Device','Room','Alert']
and the relationships ['HAS_DEVICE','TRIGGERED'], drop it safely first if it
already exists, and finally query the top devices by importance_score.

Requires the Graph Data Science plugin (installed per docker-compose). If GDS
isn't available the script prints a friendly note instead of a traceback.

Run:  python week15/kg_mastery/part2_building/04_gds_enrichment.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import get_driver, check_connection

GRAPH_NAME = "deviceGraph"
NODE_LABELS = ["Device", "Room", "Alert"]
REL_TYPES = ["HAS_DEVICE", "TRIGGERED"]


def project_graph(session):
    """(Re)create the in-memory GDS projection safely."""
    exists = session.run(
        "CALL gds.graph.exists($name) YIELD exists RETURN exists", name=GRAPH_NAME
    ).single()["exists"]
    if exists:
        session.run("CALL gds.graph.drop($name)", name=GRAPH_NAME)
        print(f"   (dropped existing projection '{GRAPH_NAME}')")

    session.run(
        "CALL gds.graph.project($name, $labels, $rels)",
        name=GRAPH_NAME, labels=NODE_LABELS, rels=REL_TYPES,
    )
    print(f"✅ Projected '{GRAPH_NAME}' over {NODE_LABELS} / {REL_TYPES}")


def compute_pagerank(session):
    """PageRank → writes importance_score."""
    session.run(
        """
        CALL gds.pageRank.write($name, {
            writeProperty: 'importance_score',
            maxIterations: 20,
            dampingFactor: 0.85
        })
        """,
        name=GRAPH_NAME,
    )
    print("✅ PageRank written to importance_score")


def detect_communities(session):
    """Louvain → writes community_id."""
    session.run(
        """
        CALL gds.louvain.write($name, {
            writeProperty: 'community_id'
        })
        """,
        name=GRAPH_NAME,
    )
    print("✅ Louvain communities written to community_id")


def generate_node2vec_embeddings(session):
    """Node2Vec → writes n2v_embedding (dim 64)."""
    session.run(
        """
        CALL gds.node2vec.write($name, {
            writeProperty: 'n2v_embedding',
            embeddingDimension: 64
        })
        """,
        name=GRAPH_NAME,
    )
    print("✅ Node2Vec embeddings (dim 64) written to n2v_embedding")


def top_devices(session, limit=10):
    rows = session.run(
        """
        MATCH (d:Device)
        WHERE d.importance_score IS NOT NULL
        RETURN d.id AS id, d.type AS type,
               d.importance_score AS score, d.community_id AS community
        ORDER BY score DESC
        LIMIT $limit
        """,
        limit=limit,
    )
    print("\nTop devices by importance_score:")
    print(f"  {'device':<16}{'type':<10}{'score':>10}  community")
    for r in rows:
        print(f"  {r['id']:<16}{(r['type'] or ''):<10}{r['score']:>10.4f}  {r['community']}")


def main():
    driver = get_driver()
    try:
        with driver.session() as s:
            try:
                # quick capability probe
                s.run("CALL gds.version() YIELD version RETURN version").single()
            except Exception as e:  # noqa: BLE001
                print("⚠️  Graph Data Science plugin not available.")
                print(f"   ({type(e).__name__}: {e})")
                print("   Ensure the GDS plugin is installed in your Neo4j container.")
                sys.exit(1)

            project_graph(s)
            compute_pagerank(s)
            detect_communities(s)
            generate_node2vec_embeddings(s)
            top_devices(s)

            # tidy up the in-memory projection
            s.run("CALL gds.graph.drop($name)", name=GRAPH_NAME)
            print(f"\n(dropped projection '{GRAPH_NAME}' — results persist as node properties)")
    finally:
        driver.close()


if __name__ == "__main__":
    if not check_connection():
        sys.exit(1)
    main()
