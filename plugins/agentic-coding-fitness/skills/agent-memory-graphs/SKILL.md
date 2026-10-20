---
name: agent-memory-graphs
description: "Teach durable agent memory using a Neo4j knowledge graph — how agents store decisions/observations as connected facts, recall them later, and answer natural-language questions via GraphRAG (NL → Cypher → answer). Covers event sourcing, the AgentMemory pattern, basic Cypher, and how shared graph memory coordinates a multi-agent system. Use when someone asks 'how do agents remember across runs?', mentions Neo4j/Cypher/knowledge graphs/GraphRAG, or is reviewing Week 14."
when_to_use: "Learner wants agents to remember across sessions, asks about Neo4j / Cypher / knowledge graphs / GraphRAG / event sourcing, or is catching up on Week 14."
---

# Agent Memory & Knowledge Graphs — Memory That Lasts (Week 14)

> **The one idea:** The `messages` list (Week 2) is *short-term* memory — it vanishes when the program ends. For agents that learn over days and coordinate with each other, you need **durable, structured memory**: a knowledge graph where every decision and observation is a connected, queryable fact.

```
Short-term (Week 2):  messages list   → gone when the script exits
Document memory (W8): vector store    → retrieve relevant TEXT
Graph memory (W14):   Neo4j graph     → store/recall structured FACTS + relationships
```

---

## Why a graph?

Agent knowledge is naturally *connected*: this **agent** PERFORMED this **event**, which INVOLVES this **room**, which HAS this **sensor**. A graph stores those relationships directly, so you can ask things a flat log can't easily answer:

- "Which rooms had more than 3 temperature overrides this month?" (predictive maintenance)
- "What did the HVAC agent decide about Room 301, and why?"
- "Which equipment shows up across the most fault events?"

Nodes = things (Agent, Event, Room, Equipment). Relationships = verbs (PERFORMED, INVOLVES). Both can carry properties.

---

## Event sourcing — every action becomes a fact

The core pattern: **don't overwrite state — append events.** Each decision/observation is a timestamped `Event` node linked to the agent and the things it touched. The full history is the memory.

```
(Agent {id})-[:PERFORMED]->(Event {type, data, ts})-[:INVOLVES]->(Room {name})
```

> 🧠 **One-line memory taxonomy:** these timestamped `Event` nodes are **episodic** memory ("*what happened, when*"). The stable entity nodes and relationships they point to (Room → HAS → Sensor) are **semantic** memory ("*facts about the world*"). Rules an agent learns to follow ("*if overrides > 3, raise a work order*") are **procedural** memory. One graph can hold all three. The deep version — embedding semantic memory, decay, and retrieval strategies — lives in the `knowledge-graph-mastery` skill (Week 15).

---

## The reusable `AgentMemory` class

A drop-in memory layer any agent can use — store, recall, and query:

```python
import json
from datetime import datetime
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

class AgentMemory:
    def __init__(self, agent_id, uri="bolt://localhost:7687", auth=("neo4j", "password")):
        self.agent_id = agent_id
        self.uri, self.auth = uri, auth
        self.driver = GraphDatabase.driver(uri, auth=auth)
        with self.driver.session() as s:               # ensure this agent exists
            s.run("MERGE (a:Agent {id:$id}) ON CREATE SET a.created=datetime()", id=agent_id)

    def store_event(self, event_type, data, entities=None):
        """Append an event linked to this agent and any domain entities."""
        eid = f"{self.agent_id}:{datetime.now().isoformat()}"
        with self.driver.session() as s:
            s.run("MATCH (a:Agent {id:$aid}) "
                  "CREATE (e:Event {id:$eid, type:$t, data:$d, ts:datetime()}) "
                  "CREATE (a)-[:PERFORMED]->(e)",
                  aid=self.agent_id, eid=eid, t=event_type, d=json.dumps(data))
            for label, name in (entities or []):        # e.g. ("Room","301")
                s.run(f"MERGE (n:{label} {{name:$name}}) "
                      f"WITH n MATCH (e:Event {{id:$eid}}) MERGE (e)-[:INVOLVES]->(n)",
                      name=name, eid=eid)
        return eid

    def recall_recent(self, event_type=None, limit=5):
        """Most recent events for this agent, newest first."""
        q = ("MATCH (a:Agent {id:$aid})-[:PERFORMED]->(e:Event) "
             + ("WHERE e.type=$t " if event_type else "")
             + "RETURN e ORDER BY e.ts DESC LIMIT $lim")
        with self.driver.session() as s:
            return [dict(r["e"]) for r in s.run(q, aid=self.agent_id, t=event_type, lim=limit)]

    def close(self):
        self.driver.close()
```

Using it:

```python
mem = AgentMemory("hvac-monitor-01")
mem.store_event("FAULT_DETECTED",
                {"room": "301", "unit": "AHU-3", "severity": "high"},
                entities=[("Room", "Room 301"), ("Equipment", "AHU-3")])
mem.store_event("FAULT_RESOLVED",
                {"room": "301", "resolution": "capacitor replaced"},
                entities=[("Room", "Room 301"), ("Equipment", "AHU-3")])

for e in mem.recall_recent():
    print(e["type"], e["data"])
```

Run it tomorrow, next week — the facts are still there. That's durable memory.

> 📁 Class repo: `week14_agent_memory_knowledge_graphs/agent_memory.py` (the full class), `week14_agent_memory_knowledge_graphs/lab1_hotel_mas.py` (a 2-agent hotel crew that reads/writes this memory), `week14_agent_memory_knowledge_graphs/lab1_hotel_memory.py`.

---

## A little Cypher (the graph query language)

Cypher is "SQL for graphs" — you draw the pattern you want with ASCII arrows.

```cypher
// Find all events involving Room 301
MATCH (e:Event)-[:INVOLVES]->(r:Room {name: "Room 301"})
RETURN e.type, e.data, e.ts
ORDER BY e.ts DESC

// Rooms with more than 3 HVAC override events (predictive maintenance signal)
MATCH (e:Event {type: "HVAC_OVERRIDE"})-[:INVOLVES]->(r:Room)
WITH r, count(e) AS overrides
WHERE overrides > 3
RETURN r.name, overrides ORDER BY overrides DESC
```

- `MATCH` = the pattern to find · `WHERE` = filter · `RETURN` = what to output · `MERGE` = create-if-absent (idempotent).

> 📁 Class repo: `week14_agent_memory_knowledge_graphs/NEO4J_TUTORIAL.md` — full Cypher walkthrough with hotel scenarios.

---

## GraphRAG — ask the graph in plain English

You don't have to write Cypher by hand. **GraphRAG** lets an LLM translate a natural-language question into Cypher, run it, and phrase the answer:

```python
from langchain_anthropic import ChatAnthropic

def build_graphrag_chain(mem, llm):
    graph = Neo4jGraph(url=mem.uri, username=mem.auth[0], password=mem.auth[1])
    graph.refresh_schema()                       # show the LLM what's in the graph
    return GraphCypherQAChain.from_llm(llm, graph=graph, allow_dangerous_requests=True)

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=512)
chain = build_graphrag_chain(mem, llm)
print(chain.invoke({"query": "What faults were detected in Room 301?"})["result"])
```

Under the hood: NL question → LLM writes Cypher → Neo4j runs it → LLM turns rows into a sentence. This is RAG (Week 8) but retrieving over a **graph** instead of a pile of text.

> ⚠️ **`allow_dangerous_requests=True` is not free.** It is mandatory here because the LLM *writes the Cypher that actually runs against your DB* — but that means a clever (or confused) question can produce a `DELETE`/`DETACH DELETE`/`MERGE` that mutates your graph. In production: **sandbox it** — connect with a **read-only Neo4j role** (so model-written queries physically cannot write), put it on a throwaway/replica DB, and add a guardrail that rejects write keywords before executing. Never point free-form GraphRAG at a production graph with write credentials. The `knowledge-graph-mastery` skill shows the safer **tool-wrapping** pattern (you write the Cypher, the LLM only fills parameters).

---

## Shared graph = multi-agent coordination
When several agents read and write the *same* graph, it becomes their shared world model. A sensor agent writes `TEMPERATURE_ALERT`; an optimizer agent reads it, decides, and writes `HVAC_SETPOINT_CHANGED`. No direct messaging needed — the graph is the blackboard. (This is the Week 14 hotel energy-optimization lab: 300 rooms, multiple agents, one graph.)

> 📁 Class repo: `week14_agent_memory_knowledge_graphs/6.hotel_kg_builder.py` (builds the hotel graph), `week14_agent_memory_knowledge_graphs/NEO4J_WEBAPP_INTEGRATION.md` (wiring a graph into a real backend).

---

## Quick start (run Neo4j locally)

```bash
docker run -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password neo4j:latest
pip install neo4j langchain-neo4j langchain-anthropic
# Browser UI at http://localhost:7474  ·  Bolt driver at bolt://localhost:7687
```

> 📁 Class repo: `week14_agent_memory_knowledge_graphs/1.docker-compose.yml`, `week14_agent_memory_knowledge_graphs/2.neo4j_connect.py`.

---

## 🧪 Guided lab (offer this)

Give an agent a memory it keeps. **No Neo4j and no API key required** — the warm-up runs on an in-memory dict so it costs **$0**; the skill drill graduates to real Neo4j if it's running.

### Warm-up (5–10 min · binary pass/fail · $0)
Prove you understand "durable memory survives a restart" without any infrastructure. Build a 20-line in-memory `AgentMemory` that mirrors the real one's API, plus a `MockLLM` so "GraphRAG" runs offline.

```python
import json
from datetime import datetime

class MemoryStore:
    """In-memory stand-in for the Neo4j AgentMemory — same store/recall API, $0."""
    def __init__(self, agent_id, db=None):
        self.agent_id = agent_id
        self.db = db if db is not None else {}          # pass a SHARED dict for multi-agent
        self.db.setdefault("events", [])

    def store_event(self, event_type, data, entities=None):
        eid = f"{self.agent_id}:{datetime.now().isoformat()}"
        self.db["events"].append({"id": eid, "agent": self.agent_id, "type": event_type,
                                  "data": data, "entities": entities or [], "ts": eid})
        return eid

    def recall_recent(self, event_type=None, limit=5):
        ev = [e for e in self.db["events"] if e["agent"] == self.agent_id]
        if event_type: ev = [e for e in ev if e["type"] == event_type]
        return list(reversed(ev))[:limit]

    def recall_related(self, label, name):              # "everything involving Room 301"
        return [e for e in self.db["events"] if (label, name) in e["entities"]]

class MockLLM:
    """Fake GraphRAG: turns recalled rows into a sentence — no API key."""
    def answer(self, question, rows):
        kinds = ", ".join(sorted({r["type"] for r in rows})) or "nothing"
        return f"Q: {question}\nA: I found {len(rows)} event(s): {kinds}."

# --- the warm-up assertions (all must pass) ---
shared = {}                                              # the "graph" both agents share
hvac = MemoryStore("hvac-01", shared)
hvac.store_event("FAULT_DETECTED", {"sev": "high"}, entities=[("Room", "301")])
hvac.store_event("FAULT_RESOLVED", {"fix": "capacitor"}, entities=[("Room", "301")])
assert len(hvac.recall_recent()) == 2                    # (1) stored + recalled

reloaded = MemoryStore("hvac-01", shared)                # NEW object = "restart"
assert len(reloaded.recall_recent()) == 2                # (2) memory SURVIVED the restart

sensor = MemoryStore("sensor-01", shared)                # second agent, same dict
sensor.store_event("TEMP_ALERT", {"c": 31}, entities=[("Room", "301")])
assert len(hvac.recall_related("Room", "301")) == 3      # (3) hvac sees the sensor's write

print(MockLLM().answer("What happened in Room 301?", hvac.recall_related("Room", "301")))
```

**Pass = all 3 asserts pass and the MockLLM prints a sentence.** If they pass, the learner has *felt* the three load-bearing ideas — persistence across "restart", recall-by-entity, and one shared store = coordination — before touching a database.

### Skill Drill (15–30 min · runnable · graduates to real Neo4j)
Swap the dict for the real graph and watch the same API light up nodes you can *see*.

1. **Start Neo4j** (Docker command above) and open the browser UI at `http://localhost:7474`.
2. **Run the real class.** Import `AgentMemory` from `week14_agent_memory_knowledge_graphs/agent_memory.py`; replay the same `store_event` calls. Refresh the browser — watch nodes/edges appear. *This is the moment it clicks.*
3. **Recall + restart.** Call `recall_recent()`, then kill the script and run a *fresh* process and recall again — the facts survived (contrast with the Week-2 `messages` list, which would be empty).
4. **GraphRAG, then make it safe.** Build the chain (`build_graphrag_chain`) and ask "what happened in Room 301?" in English; peek at the generated Cypher. Then prove the caveat above: create a **read-only Neo4j role**, reconnect, and confirm a model-written `DELETE` is *refused* by the DB.
5. **Two agents, one graph.** Spin up `AgentMemory("sensor-01")` writing alerts; show `hvac-01` can query what the sensor wrote — coordination via shared memory, no direct messaging.

**Weighted evaluation criteria (5):**
| # | Criterion | Weight |
|---|---|---|
| 1 | All 3 warm-up asserts pass on the $0 in-memory store | 1 |
| 2 | Real `AgentMemory` stores events you can *see* in the Neo4j browser | 1 |
| 3 | Memory survives a process restart (recall returns data in a fresh run) | 1 |
| 4 | GraphRAG answers in English **and** the read-only role refuses a model-written write | 1 |
| 5 | A second agent reads events the first agent wrote (shared-graph coordination) | 1 |

**Pass threshold: 4 / 5 criteria.** (Criterion 1 alone is the $0 floor — no learner should leave without it.)

**Stretch:** point them at `week14_agent_memory_knowledge_graphs/lab1_hotel_mas.py` (a 2-agent hotel crew on shared memory) and the `multi-agent-systems` skill to combine a swarm with graph memory; then the `agent-evaluation` skill to *measure* whether the GraphRAG answers are actually correct (recall vs. hallucination), and `knowledge-graph-mastery` for the production GraphRAG + RAGAS version.

End by tying the whole course together: "Week 2 gave the model a sentence of memory; Week 14 gives a team of agents a memory that outlives them."
