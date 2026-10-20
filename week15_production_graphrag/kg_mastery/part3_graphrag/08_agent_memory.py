#!/usr/bin/env python3
"""Part 3.8 — Agentic memory architecture on the graph (kg_mastery.pdf §3.8).

KEY INSIGHT: A graph is the natural store for agent memory because memory is
relational. The four memory layers map cleanly onto graph structures:

  1. WORKING memory   — the current task context (held in the prompt / runtime;
                        transient, NOT persisted here).
  2. EPISODIC memory  — a time-ordered log of what the agent DID. Stored as
                        (:Agent)-[:PERFORMED]->(:Event {type, details, ts}).
  3. SEMANTIC memory  — distilled facts the agent KNOWS, as a triple:
                        (:Entity)-[:FACT {predicate, confidence, ...}]->(:Entity).
  4. PROCEDURAL memory — learned how-to / skills (out of scope here; typically
                        stored as reusable :Procedure / tool-sequence nodes).

This is a SELF-CONTAINED class built directly on the neo4j driver (get_driver) —
no LangChain — so you can see exactly what Cypher each memory operation runs.

Requires:
  - (nothing beyond the project: neo4j driver via common.get_driver)

Run:  python week15/kg_mastery/part3_graphrag/08_agent_memory.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import check_connection, get_driver


class AgentMemory:
    """Graph-backed episodic + semantic memory for a single agent."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.driver = get_driver()
        # Ensure the agent node exists (idempotent).
        with self.driver.session() as s:
            s.run("MERGE (a:Agent {id: $id})", id=agent_id)

    def close(self):
        self.driver.close()

    # ── Episodic: log what happened ───────────────────────────────────────
    def remember_event(self, event_type: str, details: dict) -> str:
        """Append an Event to this agent's episodic timeline. Returns the event id."""
        with self.driver.session() as s:
            rec = s.run(
                """
                MATCH (a:Agent {id: $id})
                CREATE (e:Event {
                    id: randomUUID(),
                    type: $type,
                    details: $details,
                    ts: datetime()
                })
                CREATE (a)-[:PERFORMED]->(e)
                RETURN e.id AS id
                """,
                id=self.agent_id, type=event_type, details=details,
            ).single()
        return rec["id"]

    def recall_recent(self, n: int = 10) -> list:
        """Return this agent's N most recent events (newest first)."""
        with self.driver.session() as s:
            return [
                r.data()
                for r in s.run(
                    """
                    MATCH (a:Agent {id: $id})-[:PERFORMED]->(e:Event)
                    RETURN e.type AS type, e.details AS details, e.ts AS ts
                    ORDER BY e.ts DESC
                    LIMIT $n
                    """,
                    id=self.agent_id, n=n,
                )
            ]

    # ── Semantic: store a learned fact as a triple ────────────────────────
    def store_fact(self, subject: str, predicate: str, obj: str,
                   confidence: float = 1.0) -> None:
        """Store (subject)-[:FACT {predicate, confidence}]->(object) the agent learned."""
        with self.driver.session() as s:
            s.run(
                """
                MERGE (sub:Entity {name: $subject})
                MERGE (o:Entity {name: $obj})
                MERGE (sub)-[f:FACT {predicate: $predicate}]->(o)
                SET f.confidence = $confidence,
                    f.learned_by = $agent_id,
                    f.learned_at = datetime()
                """,
                subject=subject, predicate=predicate, obj=obj,
                confidence=confidence, agent_id=self.agent_id,
            )


def main():
    memory = AgentMemory("energy_optimizer_v1")
    try:
        # Episodic: log an analysis event.
        eid = memory.remember_event(
            "ANALYSIS",
            {"rooms_reviewed": "R101,R203,R305", "finding": "R305 runs hottest"},
        )
        print(f"Logged episodic event: {eid}")

        # Semantic: distil a fact with a confidence score.
        memory.store_fact("Room_301", "HAS_ISSUE", "HVAC_Overheating", 0.95)
        print("Stored semantic fact: Room_301 -HAS_ISSUE-> HVAC_Overheating (0.95)")

        # Recall the most recent episodic memories.
        print("\nRecent events:")
        for ev in memory.recall_recent(5):
            print(f"  [{ev['ts']}] {ev['type']}: {ev['details']}")
    finally:
        memory.close()


if __name__ == "__main__":
    if not check_connection():
        sys.exit(1)
    main()
