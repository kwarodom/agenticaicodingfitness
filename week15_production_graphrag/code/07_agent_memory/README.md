# 07 · Agent Memory & Knowledge Graphs (Week 14)

> Skill: `agentic-coding-fitness:agent-memory-graphs`
> **One idea:** the messages list is short-term memory that vanishes. A knowledge graph gives agents durable, structured, queryable memory that outlives the program.

### Step 0 — start Neo4j

```bash
docker compose -f docker-compose.yml up -d
# or the one-liner:
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

Open **http://localhost:7474** (login `neo4j` / `password`) so you can *watch* nodes appear.

### Step 1 — store & recall

```bash
python 01_memory_demo.py
```

Stores events (fault → work order → resolved), recalls them, adds a second
agent to the same graph, runs a Cypher count. **Rerun it** — the memory
persists (contrast with folder 01's `messages` list, which would be empty).
Refresh the Neo4j browser to see the graph grow.

### Step 2 — GraphRAG (ask in plain English)

```bash
pip install langchain-neo4j langchain-anthropic   # already in requirements.txt
python 02_graphrag.py
```

Asks questions like *"What happened in Room 301?"* — the LLM writes the Cypher,
runs it, and phrases the answer (`verbose=True` shows the generated Cypher).

- `agent_memory.py` — the reusable `AgentMemory` class (import it).
- Connection is configurable via `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` env vars.

**Two agents, one graph = coordination without messaging.** That's the Week 14 hotel
energy lab in miniature. This is the end of the course arc: Week 2 gave the model
a sentence of memory; Week 14 gives a *team* a memory that lasts.
