---
name: knowledge-graph-mastery
description: "Teach production-grade knowledge graphs and GraphRAG end to end (Week 15 kg_mastery track). Goes far beyond basic agent memory: Cypher from fundamentals to advanced (APOC, full-text, Graph Data Science algorithms), building graphs from CSV/JSON/text (LLMGraphTransformer), vector embeddings + chunking, GraphRAG across 7 frameworks (LangChain, LangGraph, CrewAI, Google ADK, FastMCP, LlamaIndex, raw driver), and — the part most people skip — EVALUATING GraphRAG with RAGAS, CI/CD gates, and production monitoring. Use when someone asks about GDS / PageRank / Louvain / Node2Vec, Text2Cypher, GraphCypherQAChain, RAGAS, evaluating a graph agent, comparing GraphRAG frameworks, or is reviewing Week 15."
when_to_use: "Learner wants to build/query a real knowledge graph (Cypher, GDS algorithms, ingestion), build GraphRAG with a specific framework (LangChain/LangGraph/CrewAI/ADK/FastMCP/LlamaIndex), measure GraphRAG quality (RAGAS, CI gate, monitoring), or is catching up on Week 15. Pick this over agent-memory-graphs when the question is about production GraphRAG, graph algorithms, framework choice, or evaluation rather than the basic 'how do agents remember' intro."
---

# Knowledge Graph Mastery — Production GraphRAG (Week 15)

> **The one idea:** Week 14 showed that a graph can be an agent's memory. Week 15 is the *mastery* track — how to **build** real graphs (ingestion + graph algorithms), **query** them (Cypher → GDS → GraphRAG across every major framework), and the step almost everyone skips: **prove it works** with RAGAS evaluation, CI/CD gates, and production monitoring.

```
Week 14 (agent-memory-graphs):  "a graph can remember"      → intro
Week 15 (this skill):           build · enrich · GraphRAG · EVALUATE → mastery
```

If the question is "how do agents remember across runs?" use **agent-memory-graphs**. Use *this* skill when the question is about graph algorithms (GDS), Text2Cypher, picking a GraphRAG framework, or measuring quality.

> 📁 Class repo: the whole track lives in `week15_production_graphrag/kg_mastery/` — six parts, each with its own runnable scripts and README. `week15_production_graphrag/kg_mastery.pdf` is the written course; `common.py` holds the shared Neo4j driver + model IDs every script imports.

---

## The hotel schema (the contract every script shares)

Everything queries one **Hotel IoT** graph, so you learn the techniques against a consistent world model:

```
(:Room {id, floor, type, capacity, rate_thb, status})
   -[:HAS_DEVICE]->  (:Device {id, type, model, manufacturer, status})
   -[:HAS_READING]-> (:SensorReading {ts, temp_c, humidity_pct, energy_kwh, occupancy})
(:Device)-[:TRIGGERED]->(:Alert {id, type, severity, message, resolved})
(:Staff)-[:PERFORMED]->(:MaintenanceJob)-[:RESOLVES]->(:Alert)
(:MaintenanceJob)-[:FOR_ROOM]->(:Room)
(:Guest)-[:STAYED_IN]->(:Room)        (:Supplier)-[:PROVIDES]->(:Device)
(:Agent)-[:PERFORMED]->(:Event {type, details, ts})   # written by agents in Parts 3–5
```

> 📁 `week15_production_graphrag/kg_mastery/common.py` documents this contract; load it once with `part1_fundamentals/00_load_hotel_dataset.py` (idempotent — safe to re-run).

---

## Part 1 — Cypher, from basics to Graph Data Science

Cypher is "SQL for graphs": you *draw* the pattern with ASCII arrows. Beyond the basics (`MATCH` / `WHERE` / `RETURN` / `MERGE`), Week 15 covers the advanced layer that turns a graph from a store into an analytics engine:

- **APOC** — utility procedures (`apoc.meta.stats()`, bulk ops, JSON import).
- **Full-text indexes** — `CREATE FULLTEXT INDEX … db.index.fulltext.queryNodes(...)` for keyword search inside the graph.
- **Graph Data Science (GDS)** — *algorithms* over a projected subgraph:

```cypher
// Project an in-memory graph, run PageRank, write scores back, then drop it
CALL gds.graph.project('hotelGraph', ['Room','Device','Alert'], ['HAS_DEVICE','TRIGGERED'])
CALL gds.pageRank.write('hotelGraph', {writeProperty:'importance_score', dampingFactor:0.85})
CALL gds.graph.drop('hotelGraph')
```

| Algorithm | Question it answers |
|---|---|
| **PageRank** | Which devices/rooms/alerts are most *central*? |
| **Louvain** | Which nodes cluster into *communities*? |
| **Node2Vec / FastRP** | A *structural embedding* per node (for ML / similarity) |
| **Betweenness** | Which nodes are critical *brokers* (supply-chain risk)? |

> 📁 `part1_fundamentals/03_cypher_advanced_gds.py` (APOC + full-text + GDS, each guarded so a missing plugin degrades gracefully), plus `01_cypher_basics.py`, `02_cypher_intermediate.py`, `04_schema_validation.py`, `05_temporal.py`, `06_multi_label.py`. Cheat sheet: `part6_reference/cypher_cheatsheet.md`.

---

## Part 2 — Building & enriching the graph

How facts *get into* the graph, three ways:

1. **Structured ingestion** — stream CSV/JSON rows via a parameterized `UNWIND` (Python-side, so it works with the Docker volume without `file:///`). → `01_load_csv.py`, `02_load_json_apoc.py`
2. **From unstructured text** — `LLMGraphTransformer` reads prose and extracts nodes + relationships automatically. → `03_llm_graph_transformer.py`
3. **Enrich what's there** — run GDS and write results back as node properties (PageRank → `importance_score`, Louvain → `community_id`, Node2Vec → `n2v_embedding`). → `04_gds_enrichment.py`

Then make it semantically searchable:

- **Vector embeddings** — embed room descriptions (OpenAI `text-embedding-3-small`, 1536 dims), store on nodes, do cosine search. → `05_vector_embeddings.py`
- **Chunking for GraphRAG** — split a document into a `(:Chunk)` graph + vector index, the foundation of hybrid graph+vector retrieval. → `06_chunking_graphrag.py`

> 📁 `week15_production_graphrag/kg_mastery/part2_building/` (+ its README has the full run table and the `:Imported` label convention so demo data never clashes with the curated dataset).

---

## Part 3 — GraphRAG across the major frameworks

**GraphRAG** = retrieve over a *graph* instead of a pile of text. The same hotel graph, queried seven different ways — each script is standalone and tells you exactly what to `pip install`:

| Framework | What it demonstrates | Script |
|---|---|---|
| **LangChain** | `GraphCypherQAChain`: NL → generated Cypher → answer (fastest path) | `01_langchain_graphcypher.py` |
| **LangGraph** | ReAct agent with read **and write** graph tools | `02_langgraph_react_agent.py` |
| **CrewAI** | Two role-based agents (Analyst → Optimizer) sharing graph tools | `03_crewai_crew.py` |
| **Google ADK** | Plain functions wrapped as `FunctionTool` (Gemini) | `04_google_adk_agent.py` |
| **FastMCP** | An MCP **server** exposing the graph as tools + a `graph://schema` resource | `05_fastmcp_server.py` |
| **LlamaIndex** | Build a KG *from documents* via triple extraction, then query | `06_llamaindex.py` |
| **Raw driver** | Hand-built Text2Cypher pipeline — the lesson is graceful error handling | `07_text2cypher_pipeline.py` |

The fastest path, `GraphCypherQAChain`:

```python
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_anthropic import ChatAnthropic
from common import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, LLM_MODEL

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
graph.refresh_schema()                       # show the LLM your live schema
chain = GraphCypherQAChain.from_llm(
    ChatAnthropic(model=LLM_MODEL, temperature=0),
    graph=graph,
    return_intermediate_steps=True,          # so you can inspect the generated Cypher
    allow_dangerous_requests=True,           # REQUIRED: the LLM emits executable Cypher
)
print(chain.invoke({"query": "Which devices triggered HIGH alerts?"})["result"])
```

> ⚠️ **`allow_dangerous_requests=True`** is mandatory because the LLM generates Cypher that runs against your DB. In production, prefer the **tool-wrapping** pattern below over free-form Cypher generation.

**The key production insight (PDF §5.1):** *put the Cypher inside the tool, hidden from the agent.* Name and describe tools like a clean API (`get_hot_rooms`, `assign_maintenance`); the model picks tools and supplies arguments but never writes a query. Safer (no arbitrary Cypher), cheaper (smaller prompts), more reliable.

**Agent memory on the graph** — four layers map onto graph structures:

| Layer | Stored as |
|---|---|
| Working | the prompt / runtime (not persisted) |
| **Episodic** | `(:Agent)-[:PERFORMED]->(:Event {type, details, ts})` |
| **Semantic** | `(:Entity)-[:FACT {predicate, confidence}]->(:Entity)` |
| Procedural | reusable `:Procedure` / tool-sequence nodes |

> 📁 `part3_graphrag/08_agent_memory.py` — a self-contained `AgentMemory` class built directly on the driver (episodic + semantic), so you see the exact Cypher each memory op runs. This is the deeper sibling of the Week-14 `AgentMemory` in the **agent-memory-graphs** skill.

---

## Part 4 — Evaluating GraphRAG (the part people skip)

A GraphRAG demo that "looks right" is not a system you can ship. Part 4 measures quality with **RAGAS** (LLM-as-judge) and keeps it from regressing.

> The general, **non-graph** version of this discipline lives in the `agent-evaluation` skill — golden datasets, LLM-as-judge bias (positional/verbosity/self-preference) and its calibration, the 5-gate CI/CD pipeline (lint → eval → cost → canary → shadow), and the DeepEval/Braintrust/Inspect-AI framework choices. Read it for the *why*; this Part 4 is the *graph-specific how*, with runnable scripts against the hotel graph.

**The four RAGAS metrics:**

| Metric | Catches | Good score |
|---|---|---|
| `faithfulness` | hallucination — claims not in the retrieved context | > 0.90 |
| `answer_relevancy` | answers that dodge the question | > 0.85 |
| `context_precision` | over-retrieval — irrelevant context pulled in | > 0.80 |
| `context_recall` | missing context — retrieval gaps | > 0.75 |

The workflow:

1. **Generate a test set** from your docs → a gold CSV. (`02_testset_generation.py`)
2. **Score** worked examples on the 4 metrics. (`01_ragas_eval.py`)
3. **Gate CI/CD** — fail the build if any metric drops below its floor (looser floors: 0.85 / 0.80 / 0.75 / 0.70), with a GitHub Actions YAML included. (`03_cicd_gate.py`)
4. **Track iterations** — the *close-the-loop* artifact: an append-only `IterationLog` (JSON) records each round's four scores + the one action you took, then prints the trend with ↑/↓ arrows so improvement (or regression) is obvious at a glance. Diagnose the lowest metric with the failure-mode taxonomy. (`05_improvement_loop.py`, `FAILURE_MODES.md`)
5. **Monitor production** — RAGAS tells you the system was good at *build* time; monitoring tells you it's still good *right now*. A `@monitored_graph_query` decorator times every query and records success / error / empty, and `get_metrics_summary()` reduces that into success rate, empty-result rate, and p99 latency you can alert on. (`04_production_monitoring.py`)

Together, `05_improvement_loop.py` (offline trend) and `04_production_monitoring.py` (live signal) close the loop: you measure, diagnose, change one thing, re-measure, and watch live traffic so regressions surface before users do.

> 📁 `week15_production_graphrag/kg_mastery/part4_evaluation/` — note its README: RAGAS's test-set API shifts between versions, so the scripts print your installed version if imports differ. Every script guards a missing key/package with a clear message instead of a traceback.

---

## Part 5 — Same toolkit, four domains

The flagship reference is the **hotel IoT agent**: a LangGraph ReAct agent with three KG tools (find hot rooms, list open alerts, assign maintenance) over the *live* loaded dataset — it can read *and write* the graph. The other three load their own small **synthetic** subgraph under a namespaced label so they never touch hotel data:

| Use case | Shows | Label |
|---|---|---|
| Hotel IoT (5.1) | ReAct agent with KG tools, on live data | *(hotel)* |
| Supply chain (5.2) | Disruption impact, alternative suppliers, Betweenness | `:SC` |
| Healthcare (5.3) | Drug-interaction detection, similar-patient cohorts | `:HC` |
| Financial fraud (5.4) | Circular money-flow ring detection, Louvain clustering | `:FIN` |

> ⚠️ Healthcare & financial use **synthetic data on purpose** — real PHI / transaction data carries HIPAA/GDPR/PDPA/AML obligations. Never load real records into a teaching graph.
> 📁 `week15_production_graphrag/kg_mastery/part5_use_cases/` · reference: `part6_reference/framework_comparison.md`.

---

## Quick start

```bash
# 1. Neo4j 5.x with APOC + Graph Data Science
docker compose -f week15_production_graphrag/kg_mastery/docker-compose.yml up -d
#    Browser UI: http://localhost:7474   (neo4j / mas_memory_2024)

# 2. Deps + keys
pip install -r week15_production_graphrag/kg_mastery/requirements.txt
#    repo-root .env:  ANTHROPIC_API_KEY (Parts 3–4) · OPENAI_API_KEY (Part 2 vectors)

# 3. Load the hotel graph FIRST — everything queries it
python week15_production_graphrag/kg_mastery/part1_fundamentals/00_load_hotel_dataset.py
```

> ⚠️ **Connection gotcha:** `common.py` defaults `NEO4J_URI` to `bolt://localhost:7690` (the Week-14 stack), but the kg_mastery `docker-compose.yml` maps Bolt to **7687**. If you use the kg_mastery compose, set `NEO4J_URI=bolt://localhost:7687` in your repo-root `.env` so the scripts connect to the right stack.

---

## 🧪 Guided lab (offer this)

Take a graph from raw data all the way to a *measured* GraphRAG system — and prove the quality gate actually blocks bad changes.

### Warm-up (5-10 min, binary pass/fail)

Run the CI/CD gate's *shape* with **no API key and no Neo4j** by porting the threshold logic out of `03_cicd_gate.py` into a $0 stub. **Pass = it prints `PASSED` on the good run and `FAILED` (exit 1) on the bad run.**

```python
# graphrag_gate_warmup.py  — $0, no key, no DB. Pure gate logic from 03_cicd_gate.py.
THRESHOLDS = {"faithfulness": 0.85, "answer_relevancy": 0.80,
              "context_precision": 0.75, "context_recall": 0.70}

def gate(scores: dict) -> bool:                 # the heart of 03_cicd_gate.py
    return all(scores[m] >= THRESHOLDS[m] for m in THRESHOLDS)

good = {"faithfulness": 0.91, "answer_relevancy": 0.88,
        "context_precision": 0.79, "context_recall": 0.82}   # iter 3 of the improvement log
bad  = {**good, "faithfulness": 0.61}                          # regression -> must fail

assert gate(good) is True,  "good scores should PASS"
assert gate(bad)  is False, "a faithfulness regression must FAIL the build"
if __name__ == "__main__":                      # guard: importing `gate` must not exit
    print("PASSED" if gate(good) else "FAILED")
    import sys; sys.exit(0 if gate(good) else 1)
```

Binary check: both asserts pass, the good run prints `PASSED`, and swapping `gate(good)`→`gate(bad)` makes it exit non-zero. That non-zero exit is exactly what fails a PR check.

### Skill Drill (15-30 min, runnable, $0 with a MockLLM stub)

Wire a fake GraphRAG pipeline into the **real** gate (`03_cicd_gate.py` already ships a `dummy_rag_function` and an inline test set, so it runs end-to-end without a live pipeline). Swap in a `MockLLM` so it needs **no `ANTHROPIC_API_KEY`**:

```python
# Drill against 03_cicd_gate.py's contract:  rag_fn(question) -> {"answer", "contexts"}
class MockLLM:
    """$0 stand-in: 'faithful' answers only repeat sentences from the context."""
    def __init__(self, faithful=True): self.faithful = faithful
    def answer(self, question, contexts):
        if self.faithful:
            return contexts[0]                       # grounded -> high faithfulness
        return "Room 999 exploded and the manager fled to Mars."  # ungrounded -> low

CTX = ["Room 305 air conditioner AC-305 triggered an HVAC fault alert on 2026-05-20."]

def make_rag_fn(faithful):
    llm = MockLLM(faithful=faithful)
    return lambda q: {"answer": llm.answer(q, CTX), "contexts": CTX}

def score(rag_fn):                                   # toy proxy for RAGAS faithfulness
    out = rag_fn("What happened to room 305?")
    grounded = out["answer"] in out["contexts"]      # claim must appear in retrieved ctx
    return {"faithfulness": 0.95 if grounded else 0.40,
            "answer_relevancy": 0.88, "context_precision": 0.80, "context_recall": 0.78}

from graphrag_gate_warmup import gate
print("faithful  ->", "PASS" if gate(score(make_rag_fn(True)))  else "FAIL")  # PASS
print("unfaithful->", "PASS" if gate(score(make_rag_fn(False))) else "FAIL")  # FAIL
```

Then, with a key/DB available, run the real thing in order — each step builds on the last:

1. **Build.** Start Neo4j, load the hotel dataset, open the Browser UI and watch the graph appear. Run `01_cypher_basics.py` and write two of your own `MATCH` queries.
2. **Enrich.** Run `04_gds_enrichment.py`; in the Browser, color nodes by `community_id` and size by `importance_score`. *See* the structure the algorithms found.
3. **Ask in English.** Wire up `GraphCypherQAChain` (`01_langchain_graphcypher.py`), ask "which devices triggered HIGH alerts?", and inspect the **generated Cypher** in the intermediate steps.
4. **Upgrade to a tool-using agent.** Run `part5_use_cases/01_hotel_iot_agent.py`; note the Cypher is hidden inside each `@tool`. Have them add a fourth tool.
5. **Now prove it works.** Run `01_ragas_eval.py` on two Q&A pairs, read the four scores, then deliberately break the retrieval and watch `faithfulness`/`context_recall` drop. This is the moment evaluation clicks.
6. **Gate it.** Wire `03_cicd_gate.py` into a GitHub Action so a regression fails the build.
7. **Close the loop.** Run `05_improvement_loop.py` to see the iteration trend, then `04_production_monitoring.py` to watch success/empty/p99 on live queries — offline trend + live signal together.

### Evaluation criteria (weighted)

| Criterion | Weight | Pass when… |
|---|---|---|
| Warm-up gate is correct | 25% | good scores `PASSED`, a faithfulness regression exits non-zero |
| MockLLM drill runs at $0 | 20% | faithful→PASS, unfaithful→FAIL with no API key/DB |
| Real gate wired | 20% | `03_cicd_gate.py` runs end-to-end (real or `dummy_rag_function`) |
| Names the failed metric | 20% | when it FAILs, they say *which* metric tripped and a likely fix (failure-mode taxonomy) |
| Closes the loop | 15% | runs `05_improvement_loop.py` + `04_production_monitoring.py` and reads the trend/alert numbers |

**Pass threshold: 4 / 5 criteria (≥ 80% by weight).** Cross-link: the general, non-graph version of this drill is in `agent-evaluation`; the framework-choice and decision context is in `models-and-patterns`.

End by tying it back: "Week 14 gave an agent a memory; Week 15 makes that memory a graph you can build at scale, query with algorithms, reason over with any framework — and *measure*, so you actually trust it in production."
