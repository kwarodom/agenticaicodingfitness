#!/usr/bin/env python3
"""Part 2.7 — Document chunking for GraphRAG (kg_mastery.pdf §2.7).

Ingests a document into a chunk graph for retrieval-augmented generation:

  (:Document)-[:HAS_CHUNK]->(:Chunk)
  (:Chunk)-[:NEXT_CHUNK]->(:Chunk)    # preserves reading order

Each chunk carries its own 1536-d embedding so you can do vector KNN over
chunks AND expand the window to neighbouring chunks for richer context.

  ingest_document(doc_text, doc_id, doc_title)
  CREATE VECTOR INDEX chunk_embeddings ON (c:Chunk) ON (c.embedding)

Demo ingests sample_data/maintenance_reports.txt.

Chunking strategy cheat-sheet (kg_mastery.pdf §2.7):

  | Strategy            | chunk_size | overlap | Best for                          |
  |---------------------|-----------:|--------:|-----------------------------------|
  | Fixed / recursive   |    256-512 |   10-20%| General prose, mixed docs (here)  |
  | Sentence            |    1-3 sent|  1 sent | QA over precise facts             |
  | Semantic            |   variable |       0 | Topic-coherent retrieval          |
  | Paragraph / section |   variable | 0-1 para| Structured reports, manuals       |
  | Token-window        |  256-1024  |   ~64tok| LLM context-budget control        |

Requires:
  - OPENAI_API_KEY in your environment / repo-root .env
  - pip install langchain-openai langchain-text-splitters

Run:  python week15/kg_mastery/part2_building/06_chunking_graphrag.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import get_driver, check_connection, EMBED_MODEL

EMBED_DIM = 1536
INDEX_NAME = "chunk_embeddings"
REPORTS_PATH = Path(__file__).resolve().parent / "sample_data" / "maintenance_reports.txt"

# --- Window retrieval with context expansion (kg_mastery.pdf §2.7) ----------
# After embedding the user's question, find the nearest chunks, then pull in the
# previous and next chunk of each hit so the LLM sees surrounding context:
#
#   CALL db.index.vector.queryNodes('chunk_embeddings', $k, $queryEmbedding)
#   YIELD node AS hit, score
#   OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(hit)
#   OPTIONAL MATCH (hit)-[:NEXT_CHUNK]->(next:Chunk)
#   RETURN hit.text AS chunk,
#          prev.text AS prev_context,
#          next.text AS next_context,
#          score
#   ORDER BY score DESC
# ----------------------------------------------------------------------------


def get_tools():
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    embedder = OpenAIEmbeddings(model=EMBED_MODEL)
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    return embedder, splitter


def create_chunk_index(session):
    session.run(
        f"""
        CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {EMBED_DIM},
            `vector.similarity_function`: 'cosine'
        }}}}
        """
    )
    print(f"✅ Vector index '{INDEX_NAME}' ready ({EMBED_DIM}d, cosine).")


def ingest_document(session, embedder, splitter, doc_text, doc_id, doc_title):
    chunks = splitter.split_text(doc_text)
    if not chunks:
        print("No chunks produced; nothing to ingest.")
        return 0

    embeddings = embedder.embed_documents(chunks)
    rows = [
        {"id": f"{doc_id}#c{i}", "text": text, "embedding": emb, "position": i}
        for i, (text, emb) in enumerate(zip(chunks, embeddings))
    ]

    session.run(
        """
        MERGE (d:Document {id: $doc_id})
        SET d.title = $doc_title

        WITH d
        UNWIND $rows AS row
        CREATE (c:Chunk {
            id: row.id, text: row.text, embedding: row.embedding,
            position: row.position, doc_id: $doc_id
        })
        MERGE (d)-[:HAS_CHUNK]->(c)

        WITH c ORDER BY c.position
        WITH collect(c) AS cs
        UNWIND range(0, size(cs) - 2) AS i
        WITH cs[i] AS prev, cs[i + 1] AS curr
        MERGE (prev)-[:NEXT_CHUNK]->(curr)
        """,
        doc_id=doc_id, doc_title=doc_title, rows=rows,
    )
    print(f"✅ Ingested '{doc_title}' as {len(rows)} chunks "
          f"(HAS_CHUNK + NEXT_CHUNK chain).")
    return len(rows)


def main():
    try:
        embedder, splitter = get_tools()
    except ImportError as e:
        print(f"⚠️  Missing dependency: {e.name}")
        print("   pip install langchain-openai langchain-text-splitters")
        sys.exit(1)

    if not REPORTS_PATH.exists():
        print(f"⚠️  Missing document: {REPORTS_PATH}")
        sys.exit(1)

    doc_text = REPORTS_PATH.read_text(encoding="utf-8")

    driver = get_driver()
    try:
        with driver.session() as s:
            create_chunk_index(s)
            ingest_document(
                s, embedder, splitter,
                doc_text=doc_text,
                doc_id="maintenance_reports",
                doc_title="Hotel Maintenance Reports",
            )
            print("\nSee the 'window retrieval with context expansion' Cypher "
                  "at the top of this file for the retrieval step.")
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
