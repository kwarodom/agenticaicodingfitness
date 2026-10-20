# Part 5 — Real-World Use Cases

Four end-to-end use cases from `kg_mastery.pdf` Part 5, showing how the same
knowledge-graph + agent toolkit applies across very different domains.

| # | File | §   | What it shows | Data |
|---|------|-----|---------------|------|
| 1 | `01_hotel_iot_agent.py`  | 5.1 | LangGraph ReAct agent with 3 KG tools (hot rooms, open alerts, assign maintenance) | **LIVE hotel dataset** |
| 2 | `02_supply_chain.py`     | 5.2 | Disruption-impact, alternative-supplier, single-supplier-risk queries | synthetic `:SC` |
| 3 | `03_healthcare.py`       | 5.3 | Drug-interaction detection, similar-patient cohorts | synthetic `:HC` |
| 4 | `04_financial_fraud.py`  | 5.4 | Circular money-flow (fraud ring) + counterparty-risk traversal | synthetic `:FIN` |

## Two kinds of script here

**The hotel agent (5.1) uses the REAL loaded dataset** — the same rooms,
devices, sensor readings, alerts, staff and maintenance jobs that Part 1/2
loaded. It is the flagship "agent with KG tools" reference: each `@tool` wraps a
hand-written Cypher query, so the LLM reasons in plain language and never writes
Cypher itself. It can also *write* to the graph (`assign_maintenance`).

**Supply-chain, healthcare and financial each load their own small SYNTHETIC
subgraph** under a namespaced label so they never touch the hotel data:

| Use case   | Label namespace | Cleanup |
|------------|-----------------|---------|
| Supply chain | `:SC`  | `MATCH (n:SC)  DETACH DELETE n;` |
| Healthcare   | `:HC`  | `MATCH (n:HC)  DETACH DELETE n;` |
| Financial    | `:FIN` | `MATCH (n:FIN) DETACH DELETE n;` |

All three loaders are **idempotent** (`MERGE` everywhere) — re-running just
updates properties, it doesn't duplicate nodes. Each script prints its own
`--clean` statement at the end.

> **Healthcare and financial use synthetic data on purpose.** Real clinical
> records are protected health information (HIPAA / GDPR / PDPA) and real
> financial records carry AML/audit obligations. The demos use only made-up,
> de-identified records so you can run them freely. Never load real PHI or real
> transaction data into a teaching graph.

## Running

```bash
# 1. Make sure Neo4j is up (docker compose in kg_mastery/)
docker compose -f week15/kg_mastery/docker-compose.yml up -d

# 2. The agent needs an API key + LangGraph stack
export ANTHROPIC_API_KEY=sk-ant-...
pip install langgraph langchain-anthropic langchain-core

# 3. Run any use case
python week15/kg_mastery/part5_use_cases/01_hotel_iot_agent.py     # live data + LLM
python week15/kg_mastery/part5_use_cases/02_supply_chain.py        # loads :SC,  Cypher only
python week15/kg_mastery/part5_use_cases/03_healthcare.py          # loads :HC,  Cypher only
python week15/kg_mastery/part5_use_cases/04_financial_fraud.py     # loads :FIN, Cypher only
```

Only `01_hotel_iot_agent.py` calls an LLM and needs `ANTHROPIC_API_KEY`. The
other three are pure Cypher references — they just need Neo4j.

## Optional: GDS algorithms

`02` and `04` include commented Graph Data Science snippets — **Betweenness
Centrality** (supply chain: which nodes are critical brokers) and **Louvain**
(financial: cluster accounts into candidate fraud rings). They require the GDS
plugin (installed via the kg_mastery `docker-compose.yml`).

## Key insight (PDF §5.1)

> Tool naming matters. Put the Cypher *inside* the tool, hidden from the agent.

Name and describe tools like a clean API (`get_hot_rooms`, `assign_maintenance`).
The model picks tools and supplies arguments; it never sees or writes the query.
That is safer (no arbitrary Cypher), cheaper (smaller prompts), and more reliable
than asking the model to generate Cypher on the fly.
