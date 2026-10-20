#!/usr/bin/env python3
"""Part 2.6 — Vector embeddings & semantic search (kg_mastery.pdf §2.6).

Embeds each Room's free-text `description` with OpenAI text-embedding-3-small
(1536 dims), stores the vector on the node, builds a native Neo4j vector index,
and runs cosine-similarity semantic search over rooms.

  embed_room_descriptions() → SET r.embedding for rooms missing one
  CREATE VECTOR INDEX room_embeddings ...
  semantic_room_search(query, k) → db.index.vector.queryNodes(...)

Demo query: "quiet room with ocean view".

Requires:
  - OPENAI_API_KEY in your environment / repo-root .env
  - pip install langchain-openai

Run:  python week15/kg_mastery/part2_building/05_vector_embeddings.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import get_driver, check_connection, EMBED_MODEL

EMBED_DIM = 1536  # text-embedding-3-small
INDEX_NAME = "room_embeddings"


def get_embedder():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=EMBED_MODEL)


def create_vector_index(session):
    session.run(
        f"""
        CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
        FOR (r:Room) ON (r.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {EMBED_DIM},
            `vector.similarity_function`: 'cosine'
        }}}}
        """
    )
    print(f"✅ Vector index '{INDEX_NAME}' ready ({EMBED_DIM}d, cosine).")


def embed_room_descriptions(session, embedder):
    """Embed rooms that have a description but no embedding yet."""
    rows = session.run(
        """
        MATCH (r:Room)
        WHERE r.embedding IS NULL AND r.description IS NOT NULL
        RETURN r.id AS id, r.description AS description
        """
    ).data()

    if not rows:
        print("All rooms with a description are already embedded. Nothing to do.")
        return 0

    ids = [r["id"] for r in rows]
    texts = [r["description"] for r in rows]
    print(f"Embedding {len(texts)} room descriptions with {EMBED_MODEL}...")
    vectors = embedder.embed_documents(texts)

    session.run(
        """
        UNWIND $rows AS row
        MATCH (r:Room {id: row.id})
        SET r.embedding = row.embedding
        """,
        rows=[{"id": i, "embedding": v} for i, v in zip(ids, vectors)],
    )
    print(f"✅ Stored embeddings on {len(ids)} rooms.")
    return len(ids)


def semantic_room_search(session, embedder, query_text, k=3):
    q_emb = embedder.embed_query(query_text)
    rows = session.run(
        """
        CALL db.index.vector.queryNodes($index, $k, $emb)
        YIELD node, score
        RETURN node.id AS id, node.type AS type,
               node.description AS description, score
        ORDER BY score DESC
        """,
        index=INDEX_NAME, k=k, emb=q_emb,
    ).data()
    print(f"\nSemantic search for: {query_text!r}")
    for r in rows:
        print(f"  [{r['score']:.3f}] {r['id']} ({r['type']}): {r['description']}")
    return rows


def main():
    try:
        embedder = get_embedder()
    except ImportError as e:
        print(f"⚠️  Missing dependency: {e.name}")
        print("   pip install langchain-openai")
        sys.exit(1)

    driver = get_driver()
    try:
        with driver.session() as s:
            create_vector_index(s)
            embed_room_descriptions(s, embedder)
            semantic_room_search(s, embedder, "quiet room with ocean view", k=3)
    finally:
        driver.close()


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set.")
        print("   export OPENAI_API_KEY=sk-...  (or add it to repo-root .env)")
        sys.exit(1)
    if not check_connection():
        sys.exit(1)
    main()
