"""A reusable, durable memory layer backed by a Neo4j knowledge graph (Week 14).

Short-term memory (the messages list, folder 01) vanishes when the program
ends. This survives: every decision/observation is an Event node, connected to
the agent and the things it touched. Event sourcing — append facts, recall later.

Connection settings come from env (with sensible defaults that match the
docker command in the folder README):
    NEO4J_URI       (default bolt://localhost:7687)
    NEO4J_USER      (default neo4j)
    NEO4J_PASSWORD  (default password)
"""
import json
import os
from datetime import datetime

from neo4j import GraphDatabase

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class AgentMemory:
    """Knowledge-graph-backed memory for any agent."""

    def __init__(self, agent_id, uri=URI, auth=(USER, PASSWORD)):
        self.agent_id = agent_id
        self.uri = uri
        self.auth = auth
        self.driver = GraphDatabase.driver(uri, auth=auth)
        with self.driver.session() as s:  # make sure this agent exists in the graph
            s.run("MERGE (a:Agent {id:$id}) ON CREATE SET a.created = datetime()", id=agent_id)

    def store_event(self, event_type, data, entities=None):
        """Append an event linked to this agent and any domain entities.

        entities: list of (Label, name) tuples, e.g. [("Room","301"),("Equipment","AHU-3")]
        """
        event_id = f"{self.agent_id}:{datetime.now().isoformat()}"
        with self.driver.session() as s:
            s.run(
                "MATCH (a:Agent {id:$aid}) "
                "CREATE (e:Event {id:$eid, type:$t, data:$d, ts:datetime()}) "
                "CREATE (a)-[:PERFORMED]->(e)",
                aid=self.agent_id, eid=event_id, t=event_type, d=json.dumps(data),
            )
            for label, name in (entities or []):
                s.run(
                    f"MERGE (n:{label} {{name:$name}}) "
                    f"WITH n MATCH (e:Event {{id:$eid}}) "
                    f"MERGE (e)-[:INVOLVES]->(n)",
                    name=name, eid=event_id,
                )
        return event_id

    def recall_recent(self, event_type=None, limit=5):
        """Most recent events for this agent, newest first."""
        q = ("MATCH (a:Agent {id:$aid})-[:PERFORMED]->(e:Event) "
             + ("WHERE e.type = $t " if event_type else "")
             + "RETURN e ORDER BY e.ts DESC LIMIT $lim")
        with self.driver.session() as s:
            return [dict(r["e"]) for r in s.run(q, aid=self.agent_id, t=event_type, lim=limit)]

    def recall_related(self, label, name, limit=10):
        """Every event this agent performed that involves a specific entity."""
        with self.driver.session() as s:
            res = s.run(
                f"MATCH (a:Agent {{id:$aid}})-[:PERFORMED]->(e:Event)"
                f"-[:INVOLVES]->(n:{label} {{name:$name}}) "
                f"RETURN e ORDER BY e.ts DESC LIMIT $lim",
                aid=self.agent_id, name=name, lim=limit,
            )
            return [dict(r["e"]) for r in res]

    def cypher(self, query, **params):
        """Run arbitrary Cypher; return a list of dicts."""
        with self.driver.session() as s:
            return [dict(r) for r in s.run(query, **params)]

    def close(self):
        self.driver.close()
