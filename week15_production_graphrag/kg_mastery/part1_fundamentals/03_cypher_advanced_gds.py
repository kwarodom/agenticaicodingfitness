#!/usr/bin/env python3
"""Part 1.5 — Advanced Cypher: APOC, Full-text, GDS, kg_mastery.pdf §1.5.

Each capability lives in its own function wrapped in try/except so that a
missing plugin degrades gracefully with a printed note instead of crashing:

  (a) APOC          — apoc.meta.stats() proves APOC is installed
  (b) Full-text     — CREATE FULLTEXT INDEX + db.index.fulltext.queryNodes
  (c) GDS           — project 'hotelGraph' over Room/Device/Alert, run
                      pageRank / louvain / fastRP, then drop the projection
  (d) Vector / KNN  — note only (covered in Part 2, needs embeddings)

Run:  python part1_fundamentals/03_cypher_advanced_gds.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import get_driver, run, check_connection

GRAPH_NAME = "hotelGraph"


def hdr(title):
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


def demo_apoc(s):
    hdr("(a) APOC — apoc.meta.stats()")
    try:
        rows = [r.data() for r in s.run("CALL apoc.meta.stats() YIELD nodeCount, relCount RETURN nodeCount, relCount")]
        stats = rows[0]
        print(f"   APOC present ✅  nodeCount={stats['nodeCount']}  relCount={stats['relCount']}")
    except Exception as e:  # noqa: BLE001
        print(f"   ⚠️  APOC unavailable: {type(e).__name__}: {e}")
        print("   Install the APOC plugin to enable this.")


def demo_fulltext(s):
    hdr("(b) Full-text index on Room(description, type, id)")
    try:
        s.run(
            "CREATE FULLTEXT INDEX room_fts IF NOT EXISTS "
            "FOR (r:Room) ON EACH [r.description, r.type, r.id]"
        )
        # newly-created full-text indexes may need a moment to come online
        s.run("CALL db.awaitIndexes(5000)")
        rows = [r.data() for r in s.run(
            """
            CALL db.index.fulltext.queryNodes('room_fts', 'ocean OR suite')
            YIELD node, score
            RETURN node.id AS id, node.type AS type, round(score, 3) AS score
            ORDER BY score DESC
            LIMIT 5
            """)]
        if rows:
            for row in rows:
                print(f"   {row['id']}  {row['type']}  score={row['score']}")
        else:
            print("   index live but no matches for 'ocean OR suite'")
    except Exception as e:  # noqa: BLE001
        print(f"   ⚠️  Full-text search failed: {type(e).__name__}: {e}")


def _gds_drop_if_exists(s):
    rows = [r.data() for r in s.run(
        "CALL gds.graph.exists($name) YIELD exists RETURN exists", name=GRAPH_NAME)]
    if rows and rows[0]["exists"]:
        s.run("CALL gds.graph.drop($name)", name=GRAPH_NAME)


def demo_gds(s):
    hdr("(c) GDS — project, pageRank / louvain / fastRP, then drop")
    try:
        _gds_drop_if_exists(s)
        # Project Room/Device/Alert with HAS_DEVICE/TRIGGERED (undirected so
        # community + embedding algos have something to traverse both ways).
        s.run(
            """
            CALL gds.graph.project(
                $name,
                ['Room', 'Device', 'Alert'],
                {
                    HAS_DEVICE: {orientation: 'UNDIRECTED'},
                    TRIGGERED:  {orientation: 'UNDIRECTED'}
                }
            )
            """, name=GRAPH_NAME)

        print("   pageRank (top 5 by score):")
        for r in s.run(
            """
            CALL gds.pageRank.stream($name)
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).id AS id, round(score, 4) AS score
            ORDER BY score DESC LIMIT 5
            """, name=GRAPH_NAME):
            row = r.data()
            print(f"     {row['id']}  pr={row['score']}")

        print("   louvain (community sizes):")
        for r in s.run(
            """
            CALL gds.louvain.stream($name)
            YIELD nodeId, communityId
            RETURN communityId, count(*) AS members
            ORDER BY members DESC LIMIT 5
            """, name=GRAPH_NAME):
            row = r.data()
            print(f"     community {row['communityId']}: {row['members']} members")

        print("   fastRP (64-dim embeddings, first node preview):")
        for r in s.run(
            """
            CALL gds.fastRP.stream($name, {embeddingDimension: 64})
            YIELD nodeId, embedding
            RETURN gds.util.asNode(nodeId).id AS id, embedding[0..4] AS preview
            ORDER BY id LIMIT 3
            """, name=GRAPH_NAME):
            row = r.data()
            preview = [round(x, 3) for x in row["preview"]]
            print(f"     {row['id']}  emb[0:4]={preview}")
    except Exception as e:  # noqa: BLE001
        print(f"   ⚠️  GDS unavailable: {type(e).__name__}: {e}")
        print("   Install the graph-data-science plugin to enable this.")
    finally:
        try:
            _gds_drop_if_exists(s)
            print("   (projection dropped)")
        except Exception:  # noqa: BLE001
            pass


def demo_vector_note():
    hdr("(d) Vector index / KNN search")
    print("   ℹ️  Native vector index KNN search is covered in Part 2 — it")
    print("   requires node embeddings (e.g. via an embedding model) before a")
    print("   VECTOR INDEX and db.index.vector.queryNodes can be used.")


def main():
    driver = get_driver()
    try:
        with driver.session() as s:
            demo_apoc(s)
            demo_fulltext(s)
            demo_gds(s)
        demo_vector_note()
        print("\n✅ Advanced demos complete.")
    finally:
        driver.close()


if __name__ == "__main__":
    if not check_connection():
        sys.exit(1)
    main()
