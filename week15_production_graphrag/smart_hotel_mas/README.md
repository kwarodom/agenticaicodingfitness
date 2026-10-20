# 🏨 Smart Hotel MAS — 4-Memory Architecture Workshop (Code Companion)

Runnable Python that follows **`week15/smart_hotel_mas.pdf`** — a 2-hour hands-on workshop building a 5-agent Multi-Agent System with a **4-layer memory architecture** over a 200-room smart hotel.

```
The 4 memory layers (different access pattern + retention per layer):
  L1 Working   → Python dict        (session.state)      [<1ms]
  L2 Episodic  → SQLite             (timestamped log)    [<10ms]
  L3 Semantic  → ChromaDB           (vector embeddings)  [50–200ms]
  L4 Knowledge → Neo4j              (structured graph)   [50–200ms]

The 5 agents:
  SensorAgent · EnergyAgent · MemoryAgent · AlertAgent · ReportAgent
```

## Layout

```
smart_hotel_mas/
├── config.py                  → shared connection settings (Neo4j, ChromaDB, model)
├── docker-compose.yml         → Neo4j + ChromaDB + Redis
├── requirements.txt
├── seed_hotel.py              → one-shot seeder (= Checkpoint 1)
├── checkpoints/               → the 6 workshop checkpoints (15 min each)
│   ├── checkpoint1_seed.py        Neo4j schema + 200 rooms / 400 devices / 2000 readings
│   ├── checkpoint2_memory.py      L1 WorkingMemory (dict) + L2 EpisodicMemory (SQLite)
│   ├── checkpoint3_semantic.py    L3 SemanticMemory (ChromaDB vector recall)
│   ├── checkpoint4_optimizer.py   PuLP HVAC optimizer + Prophet occupancy forecast
│   ├── checkpoint5_rl_anomaly.py  DQN (Stable-Baselines3) + Isolation Forest
│   └── checkpoint6_full_mas.py    L4 HotelKGMemory + full 5-agent CrewAI crew
├── patterns/                  → agent design patterns
│   ├── memory_injection.py        build context string from all 4 layers
│   └── error_handling.py          retry decorator + graceful L4→L3→L2→L1 fallback
├── graphrag/graphrag.py       → NL → Cypher → answer (LangChain + Claude)
├── production/                → prototype → production
│   ├── docker-compose.prod.yml
│   └── api/main.py                FastAPI backend (sensor batch, alerts, optimize)
└── REFERENCE.md               → math models, memory-layer perf, tool-design best practices
```

## Setup

### 1. Start the stack

```bash
docker compose -f week15/smart_hotel_mas/docker-compose.yml up -d
# Neo4j UI: http://localhost:7474  (neo4j / hotel_mas_2024)
# ChromaDB: http://localhost:8001
```

### 2. Python deps + keys

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r week15/smart_hotel_mas/requirements.txt
```

`.env` in the repo root (gitignored — never commit):

```ini
ANTHROPIC_API_KEY="sk-ant-..."   # Checkpoint 6, GraphRAG, memory injection
```

### 3. Seed the hotel (Checkpoint 1 — do this first)

```bash
python week15/smart_hotel_mas/checkpoints/checkpoint1_seed.py
# Verify in Neo4j Browser: MATCH (r:Room) RETURN count(r)  → 200
```

## Run order

Work through `checkpoints/` 1 → 6. Each is self-contained and prints what it did.
- **CP1** needs Neo4j. **CP2** is pure stdlib (dict + SQLite) — runs with no services.
- **CP3** needs ChromaDB + `chromadb` + `sentence-transformers`.
- **CP4** needs `pulp` + `prophet`. **CP5** needs `gymnasium` + `stable-baselines3` + `scikit-learn`.
- **CP6** needs Neo4j + `crewai[anthropic]` + `ANTHROPIC_API_KEY` and ties all 4 layers together. **crewai supports Python <3.14** — on 3.14 use a 3.13 venv just for CP6 (`python3.13 -m venv .venv313`).

## Notes
- Neo4j password is **`hotel_mas_2024`** (per the workshop). Override any setting via env (`NEO4J_PASSWORD`, `CHROMA_HOST`, …) — see `config.py`.
- Model: `claude-sonnet-4-6` (the PDF predates it and shows `claude-3-5-sonnet`; swap freely).
- Scripts guard optional imports / missing keys with a clear `pip install …` message instead of a traceback.
- **CP6 / crewai 1.x:** `BaseTool` now lives at `crewai.tools` (was `crewai_tools` in <1.0) and the `anthropic/<model>` LLM string needs the `crewai[anthropic]` extra — both handled by `checkpoint6_full_mas.py` and `requirements.txt`.
