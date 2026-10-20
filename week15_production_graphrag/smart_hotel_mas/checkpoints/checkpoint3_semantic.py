#!/usr/bin/env python3
"""CHECKPOINT 3 — Semantic Memory (L3, ChromaDB).

Goal: add a vector-backed memory tier so agents can recall events by *meaning*
rather than exact keys. We embed each sensor event into ChromaDB and query it
with natural language ("room temperature too high HVAC problem"), getting back
the most semantically similar past events. (smart_hotel_mas.pdf §CP3)

Run:  python week15/smart_hotel_mas/checkpoints/checkpoint3_semantic.py

Requires a running ChromaDB server (see docker-compose.yml) on CHROMA_PORT.
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import config.py
from config import CHROMA_HOST, CHROMA_PORT

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    print("Missing dependencies for semantic memory.")
    print("pip install chromadb sentence-transformers")
    sys.exit(1)


# ── L3: Semantic Memory (vector search) ──────────────────────────────────────
class SemanticMemory:
    """L3 vector memory. Embeds sensor events and recalls them by similarity.

    Uses a local SentenceTransformer ('all-MiniLM-L6-v2') for embeddings and a
    ChromaDB HTTP server for storage/search. This complements Neo4j: Cypher
    answers structured questions, ChromaDB answers fuzzy ones.
    """

    def __init__(self, host=CHROMA_HOST, port=CHROMA_PORT, collection="hotel_memory"):
        self.client = chromadb.HttpClient(host=host, port=port)
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection, embedding_function=self.embedder
        )

    def store_reading(self, room_id, reading: dict, event_type="sensor_reading"):
        doc_text = (
            f"Room {room_id}: temp={reading.get('temp_c')}°C, "
            f"humidity={reading.get('humidity')}%, "
            f"occupancy={reading.get('occupancy')}, "
            f"energy={reading.get('kwh')} kWh, "
            f"event={event_type}, time={datetime.now().strftime('%H:%M')}"
        )
        self.collection.add(
            documents=[doc_text],
            metadatas=[
                {
                    "room_id": room_id,
                    "event_type": event_type,
                    "ts": datetime.now().isoformat(),
                    **{k: str(v) for k, v in reading.items()},
                }
            ],
            ids=[f"{room_id}_{datetime.now().timestamp()}"],
        )

    def recall_similar(self, query, n_results=3, filter_room=None) -> list:
        where = {"room_id": filter_room} if filter_room else None
        res = self.collection.query(
            query_texts=[query], n_results=n_results, where=where or None
        )
        out = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for text, meta, dist in zip(docs, metas, dists):
            out.append(
                {
                    "text": text,
                    "metadata": meta,
                    "distance": dist,
                    "similarity": round(1 - dist, 3),
                }
            )
        return out

    def recall_anomalies(self, n_results=5):
        return self.recall_similar(
            "temperature spike HVAC failure high energy anomaly", n_results
        )


if __name__ == "__main__":
    try:
        mem = SemanticMemory()
    except Exception as e:  # noqa: BLE001
        print("⚠️  Cannot reach ChromaDB.")
        print(f"   ({type(e).__name__}: {e})")
        print(f"   Expected at {CHROMA_HOST}:{CHROMA_PORT}. Start it with:")
        print("   docker compose -f week15/smart_hotel_mas/docker-compose.yml up -d")
        sys.exit(1)

    print("── Storing sensor events into semantic memory ──")
    mem.store_reading(
        "R301",
        {"temp_c": 28.5, "humidity": 75, "occupancy": True, "kwh": 4.2},
        "HIGH_TEMP_ALERT",
    )
    mem.store_reading(
        "R205",
        {"temp_c": 22.0, "humidity": 55, "occupancy": False, "kwh": 1.1},
    )
    print("Stored 2 readings (R301 HIGH_TEMP_ALERT, R205 nominal).")

    print("\n── recall_similar('room temperature too high HVAC problem') ──")
    for hit in mem.recall_similar("room temperature too high HVAC problem"):
        print(f"  sim={hit['similarity']}  {hit['text']}")

    print("\n── recall_anomalies() ──")
    anomalies = mem.recall_anomalies()
    print(f"Found {len(anomalies)} anomaly-like event(s).")
    for hit in anomalies:
        print(f"  sim={hit['similarity']}  {hit['text']}")

    # ── Key Insight ─────────────────────────────────────────────────────────
    # Semantic memory answers "fuzzy" similarity questions that are impossible
    # to express in Cypher (e.g. "find readings that *feel* like an HVAC
    # failure"). It does not replace Neo4j — use it alongside the graph: the
    # graph for structured relationships, the vector store for meaning.
