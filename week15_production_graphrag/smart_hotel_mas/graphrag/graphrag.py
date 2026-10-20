#!/usr/bin/env python3
"""GraphRAG Deep Dive. (smart_hotel_mas.pdf §"GraphRAG Deep Dive")

GraphRAG = Retrieval-Augmented Generation where the "retrieval" step is a
Cypher query against the Neo4j knowledge graph (L4) instead of a vector store.
The LLM translates a natural-language question into Cypher, the graph executes
it, and the LLM phrases the rows back as an answer.

GraphRAG pipeline trace:
    question  ──▶  LLM (CYPHER_PROMPT + schema)  ──▶  Cypher query
                                                          │
                                                          ▼
                                                   Neo4j executes
                                                          │
                                                          ▼
              answer  ◀──  LLM (QA prompt + rows)  ◀──  result rows

Tip: schema injection is the #1 quality factor. A precise schema string (node
labels, key properties, relationship directions) does more for Cypher accuracy
than any prompt-engineering trick — the model cannot guess your graph shape.

Run:  python week15/smart_hotel_mas/graphrag/graphrag.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # graphrag/ -> workshop root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "checkpoints"))

from config import check_neo4j, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, MODEL

# ── Guarded optional dependencies ────────────────────────────────────────────
try:
    from langchain.prompts import PromptTemplate
except ImportError:
    try:
        from langchain_core.prompts import PromptTemplate  # newer split package
    except ImportError:
        print("GraphRAG needs LangChain. Install it with:")
        print("    pip install langchain langchain-anthropic langchain-community langchain-neo4j")
        sys.exit(1)

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    print("GraphRAG needs the Anthropic LangChain integration. Install it with:")
    print("    pip install langchain-anthropic")
    sys.exit(1)

# Neo4jGraph + chain moved between packages across LangChain versions; try the
# dedicated langchain_neo4j first, then fall back to langchain_community.
try:
    from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
except ImportError:
    try:
        from langchain_community.graphs import Neo4jGraph
        from langchain.chains import GraphCypherQAChain
    except ImportError:
        print("GraphRAG needs Neo4j graph chains. Install them with:")
        print("    pip install langchain-neo4j  # or: langchain-community langchain")
        sys.exit(1)


class HotelGraphRAG:
    """Natural-language → Cypher → answer over the hotel knowledge graph."""

    # ── Schema injection: the single biggest quality lever ───────────────────
    HOTEL_SCHEMA = """
Node labels and key properties:
  (:Room {id, floor, type, capacity, rate_thb, status})
  (:Device {id, type, room_id, model, status})
  (:SensorReading {ts, temp_c, humidity, occupancy, kwh})
  (:Alert {type, severity, threshold, value, ts})
  (:MaintenanceJob {type, priority, status, assigned_to, ts})
  (:Staff {id, name, role, shift, skills})
  (:Event {type, details, ts})

Relationship types:
  (:Room)-[:HAS_DEVICE]->(:Device)
  (:Device)-[:RECORDED]->(:SensorReading)
  (:Alert)-[:AFFECTS]->(:Room)
  (:MaintenanceJob)-[:ASSIGNED_TO]->(:Staff)
  (:Agent)-[:PERFORMED]->(:Event)
""".strip()

    # ── Cypher-generation prompt with the 5 rules from the PDF ───────────────
    CYPHER_PROMPT = PromptTemplate(
        input_variables=["schema", "question"],
        template="""You are a Neo4j Cypher expert for a smart-hotel knowledge graph.

Graph schema:
{schema}

Rules:
1. ALWAYS add LIMIT 20 to every query unless an aggregation makes it unnecessary.
2. Use toString() when returning datetime properties so they render cleanly.
3. Use avg(), sum(), or count() for aggregation questions.
4. For time filters use: datetime() - duration({{hours: N}}).
5. Use meaningful aliases (room, device, reading, alert, staff) — never single letters.

Question: {question}

Return ONLY the Cypher query, no prose, no markdown fences.""",
    )

    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD
        )
        self.llm = ChatAnthropic(model=MODEL, temperature=0)
        self.chain = GraphCypherQAChain.from_llm(
            self.llm,
            graph=self.graph,
            cypher_prompt=self.CYPHER_PROMPT,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
            top_k=10,
        )

    def query(self, question: str) -> dict:
        """Run a NL question through the GraphRAG chain.

        Returns {answer, cypher, kg_result}. ``cypher`` and ``kg_result`` are
        pulled from the chain's intermediate steps when available.
        """
        result = self.chain.invoke({"query": question})
        answer = result.get("result", "")
        steps = result.get("intermediate_steps", []) or []
        cypher = ""
        kg_result = None
        for step in steps:
            if isinstance(step, dict):
                if "query" in step:
                    cypher = step["query"]
                if "context" in step:
                    kg_result = step["context"]
        return {"answer": answer, "cypher": cypher, "kg_result": kg_result}


if __name__ == "__main__":
    if not check_neo4j():
        sys.exit(1)
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.")
        print("    export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    rag = HotelGraphRAG()

    questions = [
        "Which rooms had temperature above 28°C today?",
        "What is the average energy consumption per floor?",
        "Which staff members are available for HVAC repairs?",
        "How many alerts were triggered in the last 4 hours?",
    ]

    for q in questions:
        print("\n" + "═" * 70)
        print(f"Q: {q}")
        try:
            out = rag.query(q)
            print(f"  Cypher: {out['cypher'][:80]}")
            print(f"  Answer: {out['answer'][:150]}")
        except Exception as e:  # noqa: BLE001
            print(f"  (query failed: {type(e).__name__}: {e})")

    # Reminder: schema injection is the #1 quality factor for GraphRAG accuracy.
