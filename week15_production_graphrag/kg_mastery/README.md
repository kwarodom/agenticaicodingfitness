# 🕸️ Knowledge Graph Mastery — Code Companion

Runnable Python that follows **`week15/kg_mastery.pdf`** step by step — from Neo4j/Cypher fundamentals to production GraphRAG with every major framework, plus RAGAS evaluation.

```
kg_mastery/
├── common.py                  → shared Neo4j driver + the hotel schema contract
├── docker-compose.yml         → Neo4j 5.x with APOC + Graph Data Science
├── requirements.txt
├── part1_fundamentals/        → Cypher basics → intermediate → advanced (GDS), schema, temporal
├── part2_building/            → load CSV/JSON, LLMGraphTransformer, GDS enrichment, vectors, chunking
├── part3_graphrag/            → LangChain · LangGraph · CrewAI · Google ADK · FastMCP · LlamaIndex · AgentMemory
├── part4_evaluation/          → RAGAS metrics, test generation, CI/CD gate, monitoring
├── part5_use_cases/           → Hotel IoT agent · Supply Chain · Healthcare · Financial fraud
└── part6_reference/           → Cypher cheat sheet + framework/tool selection
```

## Setup

### 1. Start Neo4j (with APOC + GDS)

```bash
docker compose -f week15/kg_mastery/docker-compose.yml up -d
# Browser UI: http://localhost:7474   (login: neo4j / mas_memory_2024)
# Bolt:       bolt://localhost:7687
```

### 2. Python deps + keys

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r week15/kg_mastery/requirements.txt
```

Add keys to a `.env` in the **repo root** (gitignored — never commit):

```ini
ANTHROPIC_API_KEY="sk-ant-..."     # Parts 3 & 4 (LLM, GraphRAG, RAGAS)
OPENAI_API_KEY="sk-..."            # Part 2 vector embeddings (text-embedding-3-small)
```

### 3. Load the hotel dataset (do this first!)

Everything queries the **hotel IoT** graph used throughout the course:

```bash
python week15/kg_mastery/part1_fundamentals/00_load_hotel_dataset.py
```

This MERGEs rooms, devices, sensor readings, alerts, staff, maintenance jobs, guests and suppliers — idempotent, safe to re-run.

## The hotel schema (the contract every script uses)

```
(:Room {id, floor, type, capacity, rate_thb, status, description})
   -[:HAS_DEVICE]->   (:Device {id, type, model, manufacturer, installed_at, status})
   -[:HAS_READING]->  (:SensorReading {ts, temp_c, humidity_pct, energy_kwh, occupancy})
(:Device)-[:TRIGGERED]->(:Alert {id, type, severity, message, ts, resolved})
(:Staff {id, name, role, shift})-[:PERFORMED]->(:MaintenanceJob {id, type, started_at, completed_at, status})
(:MaintenanceJob)-[:RESOLVES]->(:Alert)
(:MaintenanceJob)-[:FOR_ROOM]->(:Room)
(:Staff)-[:ASSIGNED_TO]->(:Alert)
(:Guest {id, name, check_in, check_out})-[:STAYED_IN]->(:Room)
(:Supplier {id, name, country, category})-[:PROVIDES]->(:Device)
(:Agent {id})-[:PERFORMED]->(:Event {id, type, details, ts})   # written by agents in Parts 3–5
```

> Model IDs used: `claude-sonnet-4-6` (reasoning) and `claude-haiku-4-5-20251001` (fast). Swap freely.

## Order to run

1. `part1_fundamentals/` — load dataset, then run the Cypher tours.
2. `part2_building/` — alternative ingestion + graph enrichment.
3. `part3_graphrag/` — pick a framework; each is standalone.
4. `part4_evaluation/` — measure your GraphRAG with RAGAS.
5. `part5_use_cases/` — full reference agents per domain.

Each folder has its own README. Scripts that need Neo4j / a key fail gracefully with setup instructions.
