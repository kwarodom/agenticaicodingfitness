# 🏨 Smart Hotel MAS — Build & Run Summary

A record of standing up the **Smart Hotel Multi-Agent System** workshop end-to-end on a local machine: what was built, what was verified working, and where to take it next.

> Scope: the 6-checkpoint workshop in `week15/smart_hotel_mas/` — a 5-agent MAS with a 4-layer memory architecture over a 200-room smart hotel. All six checkpoints were run and verified.

---

## 1. What we built

A **multi-agent system** where five specialized agents coordinate over **four distinct memory layers**, each chosen for a different access pattern and retention need.

### The 4-layer memory architecture

| Layer | Store | Holds | Latency |
|-------|-------|-------|---------|
| **L1 Working** | Python dict (`session.state`) | current task context | <1 ms |
| **L2 Episodic** | SQLite | timestamped log of what happened | <10 ms |
| **L3 Semantic** | ChromaDB (vector) | embeddings for similarity/anomaly recall | 50–200 ms |
| **L4 Knowledge** | Neo4j (graph) | structured entities + relationships | 50–200 ms |

### The 5 agents (CrewAI, sequential process)

| Agent | Role | Tool(s) |
|-------|------|---------|
| **SensorAgent** | Read live room sensors, surface anomalies | `read_sensors` |
| **EnergyAgent** | Compute HVAC setpoints balancing comfort vs. kWh | `optimize_hvac` (writes L4) |
| **MemoryAgent** | Recall recent activity from the graph | `query_knowledge_graph` (GraphRAG) |
| **AlertAgent** | Triage and prioritise active alerts | `query_knowledge_graph` |
| **ReportAgent** | Write the operations briefing from graph data | `query_knowledge_graph` |

The graph (L4) is the **shared blackboard**: one agent's writes (e.g. an HVAC optimization, an anomaly alert) are immediately queryable by every other agent — no direct messaging required.

```
SensorAgent ─▶ EnergyAgent ─▶ MemoryAgent ─▶ AlertAgent ─▶ ReportAgent
      │             │ writes        │ reads        │ reads        │ reads
      └─────────────┴──────── shared Neo4j L4 graph ─────────────┘
                    (+ L1 dict · L2 SQLite · L3 ChromaDB)
```

---

## 2. Infrastructure that was stood up

**Docker stack** (`docker-compose.local.yml` — conflict-free host ports chosen for this machine, since 7474/8001/6379 were already taken):

| Service | Container | Host port | Notes |
|---------|-----------|-----------|-------|
| Neo4j 5.18 (+APOC, GDS) | `smart-hotel-neo4j` | UI **7475**, Bolt **7687** | auth `neo4j` / `hotel_mas_2024`, heap capped 1 GB |
| ChromaDB | `smart-hotel-chromadb` | **8002** | HTTP API |
| Redis | `smart-hotel-redis` | **6383** | not used by the checkpoints |

**Python environments** (crewai supports Python <3.14, so two venvs):

| venv | Python | Used for |
|------|--------|----------|
| `.venv` | 3.14 | CP1–CP5 |
| `.venv313` | 3.13 | CP6 (crewai) |

**Config** — `week15/smart_hotel_mas/.env` (gitignored) points the scripts at this stack (`NEO4J_URI=bolt://localhost:7687`, `CHROMA_PORT=8002`, `hotel_mas_2024`, `ANTHROPIC_API_KEY`). Because `config.py`'s `find_dotenv()` resolves from its own directory, this local `.env` wins for the workshop while the repo-root `.env` keeps serving the separate `kg_mastery` workshop.

---

## 3. What we achieved — checkpoint by checkpoint

| CP | Topic | Result (verified) |
|----|-------|--------------------|
| **1** | Seed the graph | ✅ **200 rooms / 400 devices / 2000 sensor readings** loaded into Neo4j, isolated on 7687 |
| **2** | L1 + L2 memory | ✅ Working-memory dict + SQLite episodic log; recalled the last hour's events |
| **3** | L3 semantic | ✅ ChromaDB + MiniLM embeddings; `recall_similar("room temperature too high HVAC problem")` and `recall_anomalies()` returned ranked matches |
| **4** | Optimize + forecast | ✅ Prophet occupancy forecast feeding a **PuLP** linear program → per-room HVAC setpoints |
| **5** | RL + anomaly | ✅ **DQN** (Stable-Baselines3) trained 10k steps; **Isolation Forest** fit on 2000 real rows flagged a 38.5 °C reading as anomalous |
| **6** | Full MAS (L4 + crew) | ✅ `HotelKGMemory` (L4) write/recall + GraphRAG queries; **5-agent CrewAI crew ran end-to-end** and produced a two-paragraph operations briefing |

### The moment it all came together (CP6)
The crew's final report **surfaced a `HIGH`-severity `SENSOR_ANOMALY` for room R301** that had been written to the graph earlier by a *different* code path — proving the L4 graph works as genuine cross-agent shared memory. Meanwhile the `EnergyAgent`'s `optimize_hvac` tool wrote fresh `hvac_optimization` events back into the same graph during the run.

---

## 4. Key lessons & insights

- **Hide the query inside the tool.** Every agent tool (`get_hot_rooms`, `optimize_hvac`, `query_knowledge_graph`) wraps hand-written Cypher; the LLM reasons in plain language and never emits a query. Safer (no arbitrary Cypher), cheaper (smaller prompts), more reliable.
- **The graph is the coordination layer.** Five agents stay in sync by reading/writing one Neo4j graph rather than passing messages — the blackboard pattern.
- **Right store for the right memory.** Sub-millisecond task state (dict) and relationship reasoning (graph) are different problems; forcing them into one store is the common anti-pattern this architecture avoids.
- **Pin your agent-framework Python.** `crewai` supports Python `>=3.10,<3.14`; on a 3.14 default we needed a dedicated 3.13 venv for CP6 only.

### Compatibility fix shipped this session (PR #16)
The workshop's `checkpoint6_full_mas.py` predated crewai 1.x. Two breakages were fixed so it runs on current releases:
1. `BaseTool` moved `crewai_tools` → `crewai.tools` (now a version-tolerant import).
2. The `anthropic/<model>` LLM string needs the `crewai[anthropic]` extra (now in `requirements.txt`).

---

## 5. How to reproduce

```bash
# 1. Stack
docker compose -f week15/smart_hotel_mas/docker-compose.local.yml up -d

# 2. CP1–CP5  (Python 3.14 OK)
python3 -m venv week15/smart_hotel_mas/.venv
week15/smart_hotel_mas/.venv/bin/pip install -r week15/smart_hotel_mas/requirements.txt
for cp in 1_seed 2_memory 3_semantic 4_optimizer 5_rl_anomaly; do
  week15/smart_hotel_mas/.venv/bin/python week15/smart_hotel_mas/checkpoints/checkpoint${cp}.py
done

# 3. CP6  (needs Python <3.14 for crewai)
python3.13 -m venv week15/smart_hotel_mas/.venv313
week15/smart_hotel_mas/.venv313/bin/pip install -r week15/smart_hotel_mas/requirements.txt
week15/smart_hotel_mas/.venv313/bin/python week15/smart_hotel_mas/checkpoints/checkpoint6_full_mas.py

# teardown
docker compose -f week15/smart_hotel_mas/docker-compose.local.yml down
```

> Requires `ANTHROPIC_API_KEY` (CP6, GraphRAG) in `week15/smart_hotel_mas/.env`.

---

## 6. Further improvements

### Architecture & reliability
- **Wire Redis in (L2.5).** Redis is in the stack but unused. Use it for fast cross-process working memory / pub-sub so agents can run as separate services instead of one sequential crew.
- **Close the memory-fallback loop.** `patterns/error_handling.py` defines an L4→L3→L2→L1 graceful degradation path — exercise it in CP6 so a Neo4j outage downgrades to semantic/episodic recall instead of failing.
- **Make writes idempotent + transactional.** `optimize_hvac` and anomaly alerts currently `CREATE` on every run; switch to `MERGE` keyed by room+timestamp to avoid duplicate events on re-runs.
- **Resolve alerts.** The schema models `(:Alert)-[:RESOLVED_BY]->(:MaintenanceJob)` but the crew only raises alerts — add an agent/tool that closes the loop and writes resolutions.

### Intelligence
- **Real GraphRAG over hand-routed keywords.** `query_knowledge_graph` matches on substrings (`"alert"`, `"floor"`). Replace with a `GraphCypherQAChain` (NL → generated Cypher) for open-ended questions — see the `kg_mastery` Part 3 companion.
- **Feed real sensor data to the agents.** `read_sensors` returns simulated `gauss()` values; point it at the 2000 seeded `SensorReading` rows so the agents reason over the actual graph.
- **Use the trained models in the loop.** The DQN policy (CP5) and Prophet forecast (CP4) are trained but not yet called by `EnergyAgent` — have `optimize_hvac` invoke them instead of `random` setpoints.
- **Add GDS-driven insight.** Run PageRank/Louvain (the plugins are installed) to find the most failure-central devices or cluster rooms by behaviour, and expose it as an agent tool.

### Evaluation & operations
- **Evaluate the crew with RAGAS.** Score the ReportAgent's briefings for faithfulness/relevancy against the graph (the `kg_mastery` Part 4 companion has the harness + CI gate).
- **Add observability.** crewai tracing is available (was auto-disabled this run); enable it plus the `production/api/main.py` FastAPI metrics for latency/cost/empty-result tracking.
- **Memory & cost controls.** Three local Neo4j stacks exhausted an 8 GB Docker (one was OOM-killed). For a real deployment, give each stack a memory cap and run only what's active; meter Claude token usage per agent.

### Productionization
- **Promote from prototype.** `production/docker-compose.prod.yml` + `production/api/main.py` exist — wire the crew behind the FastAPI endpoints (sensor batch, alerts, optimize) and run the agents on a schedule or event trigger rather than one-shot.
- **Persist agent identity & audit.** Write each agent's actions as `(:Agent)-[:PERFORMED]->(:Event)` consistently so the graph doubles as an audit trail.

---

*Built and verified locally across CP1–CP6; crewai-1.x compatibility fix merged in PR #16.*
