# Part 2 — Building Knowledge Graphs

Runnable companions to **`kg_mastery.pdf` Part 2 (Building KGs)**: structured
ingestion, LLM extraction from text, graph enrichment with GDS, vector
embeddings, and document chunking for GraphRAG.

> Load the hotel dataset first: `python ../part1_fundamentals/00_load_hotel_dataset.py`

## Order to run

| # | Script | PDF § | What it does | Needs |
|---|--------|-------|--------------|-------|
| 1 | `01_load_csv.py` | 2.1 | Load `rooms.csv` → `:Room:Imported` | Neo4j |
| 2 | `02_load_json_apoc.py` | 2.1 | Load `iot_readings.json` → `:SensorReading` | Neo4j (APOC) |
| 3 | `03_llm_graph_transformer.py` | 2.2 | Text → graph via `LLMGraphTransformer` | `ANTHROPIC_API_KEY`, `langchain-experimental` |
| 4 | `04_gds_enrichment.py` | 2.5 | PageRank · Louvain · Node2Vec | Neo4j (GDS plugin) |
| 5 | `05_vector_embeddings.py` | 2.6 | Embed room descriptions + semantic search | `OPENAI_API_KEY`, `langchain-openai` |
| 6 | `06_chunking_graphrag.py` | 2.7 | Chunk a doc into a `:Chunk` graph + vector index | `OPENAI_API_KEY`, `langchain-openai`, `langchain-text-splitters` |

## Notes

- **CSV/JSON loaders use Python-side `UNWIND`, not `file:///`.** Neo4j's
  `LOAD CSV FROM 'file:///...'` and `apoc.load.json('file:///...')` require the
  file inside the server's `import/` dir, which our docker volume doesn't expose.
  Scripts 1 and 2 read the file in Python and stream rows via a parameterized
  `UNWIND` — same result, works with the docker setup. The raw Cypher is shown
  in each file's comment block for reference.
- **Sample data** lives in `sample_data/`. Imported rooms use `X1xx` ids and an
  extra `:Imported` label so they never clash with the curated `R1xx` dataset.
  Clean up with: `MATCH (r:Room:Imported) DETACH DELETE r`.
- **LLM-extracted nodes (script 3) are separate** from the curated hotel graph —
  their ids come from the model reading the text.
- **Keys** go in a repo-root `.env` (see top-level README). Scripts guard missing
  keys / packages with a clear message instead of a traceback.

## Dependencies

```bash
pip install langchain-experimental langchain-neo4j langchain-anthropic \
            langchain-core langchain-openai langchain-text-splitters
```

(All also covered by `../requirements.txt`.) The GDS and APOC plugins ship with
the course `docker-compose.yml`.
