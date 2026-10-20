"""agent_memory.py — Reusable AgentMemory for all MAS labs.

Drop-in memory layer for any agent in this repo:

    from week14.agent_memory import AgentMemory          # from repo root
    # or copy file next to your agent and import directly

    mem = AgentMemory("my-agent-id")
    mem.store_event("DECISION", {"action": "open_valve"}, entities=[("Room", "301")])
    mem.recall_recent()
    mem.recall_related("Room", "301")
    chain = mem.build_graphrag_chain(llm)  # NL → Cypher → answer
    mem.close()
"""
import json
from datetime import datetime
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "mas_memory_2024")


class AgentMemory:
    """Knowledge-graph-backed memory for any MAS agent."""

    def __init__(self, agent_id: str, uri: str = URI, auth: tuple = AUTH):
        self.agent_id = agent_id
        self.uri = uri
        self.auth = auth
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self._ensure_agent_node()

    # ── Internal ──────────────────────────────────────────────────────────────
    def _ensure_agent_node(self):
        with self.driver.session() as s:
            s.run(
                "MERGE (a:Agent {id: $id}) ON CREATE SET a.created = datetime()",
                id=self.agent_id,
            )

    # ── Store ─────────────────────────────────────────────────────────────────
    def store_event(self, event_type: str, data: dict, entities: list = None) -> str:
        """Write an event node linked to this agent.

        Args:
            event_type: e.g. "FAULT_DETECTED", "DECISION", "OBSERVATION"
            data:       arbitrary dict payload (serialised as JSON)
            entities:   list of (Label, name) tuples for domain nodes to link
                        e.g. [("Room", "301"), ("Equipment", "AHU-3")]
        Returns:
            event_id string
        """
        event_id = f"{self.agent_id}:{datetime.now().isoformat()}"
        with self.driver.session() as s:
            s.run(
                "MATCH (a:Agent {id: $aid}) "
                "CREATE (e:Event {id: $eid, type: $etype, data: $data, ts: datetime()}) "
                "CREATE (a)-[:PERFORMED]->(e)",
                aid=self.agent_id,
                eid=event_id,
                etype=event_type,
                data=json.dumps(data),
            )
            for label, name in (entities or []):
                s.run(
                    f"MERGE (n:{label} {{name: $name}}) "
                    f"WITH n MATCH (e:Event {{id: $eid}}) "
                    f"MERGE (e)-[:INVOLVES]->(n)",
                    name=name,
                    eid=event_id,
                )
        return event_id

    # ── Recall ────────────────────────────────────────────────────────────────
    def recall_recent(self, event_type: str = None, limit: int = 5) -> list:
        """Return the most recent events for this agent, newest first."""
        with self.driver.session() as s:
            if event_type:
                result = s.run(
                    "MATCH (a:Agent {id: $aid})-[:PERFORMED]->(e:Event) "
                    "WHERE e.type = $etype "
                    "RETURN e ORDER BY e.ts DESC LIMIT $lim",
                    aid=self.agent_id, etype=event_type, lim=limit,
                )
            else:
                result = s.run(
                    "MATCH (a:Agent {id: $aid})-[:PERFORMED]->(e:Event) "
                    "RETURN e ORDER BY e.ts DESC LIMIT $lim",
                    aid=self.agent_id, lim=limit,
                )
            return [dict(r["e"]) for r in result]

    def recall_related(self, entity_label: str, entity_name: str, limit: int = 10) -> list:
        """Return all events this agent performed that involve a specific entity."""
        with self.driver.session() as s:
            result = s.run(
                f"MATCH (a:Agent {{id: $aid}})-[:PERFORMED]->(e:Event)"
                f"-[:INVOLVES]->(n:{entity_label} {{name: $name}}) "
                f"RETURN e ORDER BY e.ts DESC LIMIT $lim",
                aid=self.agent_id, name=entity_name, lim=limit,
            )
            return [dict(r["e"]) for r in result]

    def recall_all_agents(self) -> list:
        """Return every agent node in the graph (useful for MAS oversight)."""
        return self.cypher("MATCH (a:Agent) RETURN a.id AS id, a.created AS created")

    # ── GraphRAG ──────────────────────────────────────────────────────────────
    def build_graphrag_chain(self, llm, verbose: bool = False):
        """Return a GraphCypherQAChain that answers NL questions about this graph.

        Usage:
            chain = mem.build_graphrag_chain(llm)
            answer = chain.invoke({"query": "What faults occurred in Room 301?"})
            print(answer["result"])
        """
        graph = Neo4jGraph(url=self.uri, username=self.auth[0], password=self.auth[1])
        graph.refresh_schema()
        # Append sample event types so the LLM generates correct Cypher
        event_types = self.cypher("MATCH (e:Event) RETURN DISTINCT e.type AS t")
        if event_types:
            types_str = ", ".join(f'"{r["t"]}"' for r in event_types if r["t"])
            graph.schema += f"\n\nKnown Event.type values: {types_str}"
        return GraphCypherQAChain.from_llm(
            llm, graph=graph, verbose=verbose, allow_dangerous_requests=True,
        )

    # ── Raw access ────────────────────────────────────────────────────────────
    def cypher(self, query: str, **params) -> list:
        """Run arbitrary Cypher and return a list of dicts."""
        with self.driver.session() as s:
            return [dict(r) for r in s.run(query, **params)]

    def close(self):
        self.driver.close()


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from langchain_anthropic import ChatAnthropic

    load_dotenv()

    # ── 1. Basic memory (no LLM needed) ───────────────────────────────────────
    print("=== HVAC Monitor Agent ===")
    mem = AgentMemory("hvac-monitor-agent-01")

    mem.store_event(
        "FAULT_DETECTED",
        {"room": "301", "unit": "AHU-3", "severity": "high"},
        entities=[("Room", "Room 301"), ("Equipment", "AHU-3")],
    )
    mem.store_event(
        "WORK_ORDER_CREATED",
        {"work_order_id": "WO-9921", "assigned_to": "Maintenance Team A"},
        entities=[("Equipment", "AHU-3")],
    )
    mem.store_event(
        "FAULT_RESOLVED",
        {"room": "301", "unit": "AHU-3", "resolution": "capacitor replaced"},
        entities=[("Room", "Room 301"), ("Equipment", "AHU-3")],
    )

    print("\n--- Recent events (all types) ---")
    for e in mem.recall_recent(limit=5):
        print(f"  [{e['type']}] {e['data']}")

    print("\n--- Only FAULT_DETECTED events ---")
    for e in mem.recall_recent(event_type="FAULT_DETECTED"):
        print(f"  [{e['type']}] {e['data']}")

    print("\n--- Events involving AHU-3 ---")
    for e in mem.recall_related("Equipment", "AHU-3"):
        print(f"  [{e['type']}] {e['data']}")

    # ── 2. Multi-agent: second agent stores its own events ─────────────────────
    print("\n=== Sensor Agent ===")
    sensor = AgentMemory("sensor-agent-01")
    sensor.store_event(
        "TEMPERATURE_ALERT",
        {"room": "301", "temp": 29.5, "threshold": 26.0},
        entities=[("Room", "Room 301")],
    )
    sensor.store_event(
        "OBSERVATION",
        {"room": "301", "co2_ppm": 1200, "status": "elevated"},
        entities=[("Room", "Room 301")],
    )

    print("\n--- All agents in graph ---")
    for row in mem.recall_all_agents():
        print(f"  {row}")

    # ── 3. GraphRAG — NL → Cypher → answer ────────────────────────────────────
    print("\n=== GraphRAG Query ===")
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_tokens=512,
    )
    chain = mem.build_graphrag_chain(llm, verbose=True)

    for question in [
        "What faults were detected?",
        "Which equipment was involved in work orders?",
        "What happened in Room 301?",
    ]:
        print(f"\nQ: {question}")
        answer = chain.invoke({"query": question})
        print(f"A: {answer['result']}")

    mem.close()
    sensor.close()
    print("\nDone.")
