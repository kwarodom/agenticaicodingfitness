#!/usr/bin/env python3
"""Part 2.6 (LOCAL variant) — Vector embeddings & semantic search, no API key.

Same lesson as 05_vector_embeddings.py, but swaps OpenAI's text-embedding-3-small
for a LOCAL sentence-transformers model (all-MiniLM-L6-v2, 384 dims). Runs fully
offline on CPU — no OPENAI_API_KEY, no network calls after the first model
download.

Why local instead of "Anthropic embeddings"? Anthropic has no embeddings API
(Claude is text-generation only; Anthropic points users to Voyage AI). So for a
key-free, offline path, a local model is the right choice.

To avoid colliding with the OpenAI version (1536 dims), this script stores its
vector on a SEPARATE property `r.embedding_local` behind its own index
`room_embeddings_local`. The two can coexist on the same nodes.

  embed_room_descriptions() → SET r.embedding_local for rooms missing one
  CREATE VECTOR INDEX room_embeddings_local ...
  semantic_room_search(query, k) → db.index.vector.queryNodes(...)

Requires:  pip install sentence-transformers   (pulls in torch)
Run:       python week15/kg_mastery/part2_building/05_vector_embeddings_local.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import get_driver, check_connection

LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # all-MiniLM-L6-v2 output size
INDEX_NAME = "room_embeddings_local"
EMBED_PROP = "embedding_local"


class LocalEmbedder:
    """Tiny wrapper exposing the same embed_documents/embed_query interface the
    lesson uses, backed by a local sentence-transformers model. Vectors are
    L2-normalised so cosine similarity behaves well."""

    def __init__(self, model_name=LOCAL_MODEL):
        from sentence_transformers import SentenceTransformer

        print(f"Loading local model {model_name} (first run downloads ~80MB)...")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()


def create_vector_index(session):
    session.run(
        f"""
        CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
        FOR (r:Room) ON (r.{EMBED_PROP})
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {EMBED_DIM},
            `vector.similarity_function`: 'cosine'
        }}}}
        """
    )
    print(f"✅ Vector index '{INDEX_NAME}' ready ({EMBED_DIM}d, cosine).")


def embed_room_descriptions(session, embedder):
    """Embed rooms that have a description but no local embedding yet."""
    rows = session.run(
        f"""
        MATCH (r:Room)
        WHERE r.{EMBED_PROP} IS NULL AND r.description IS NOT NULL
        RETURN r.id AS id, r.description AS description
        """
    ).data()

    if not rows:
        print("All rooms with a description are already embedded. Nothing to do.")
        return 0

    ids = [r["id"] for r in rows]
    texts = [r["description"] for r in rows]
    print(f"Embedding {len(texts)} room descriptions with {LOCAL_MODEL}...")
    vectors = embedder.embed_documents(texts)

    session.run(
        f"""
        UNWIND $rows AS row
        MATCH (r:Room {{id: row.id}})
        SET r.{EMBED_PROP} = row.embedding
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
        embedder = LocalEmbedder()
    except ImportError as e:
        print(f"⚠️  Missing dependency: {e.name}")
        print("   pip install sentence-transformers")
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
    if not check_connection():
        sys.exit(1)
    main()
