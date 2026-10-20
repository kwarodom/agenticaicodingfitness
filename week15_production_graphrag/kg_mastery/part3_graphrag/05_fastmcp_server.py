#!/usr/bin/env python3
"""Part 3.5 — FastMCP server exposing the graph (kg_mastery.pdf §3.5).

KEY INSIGHT: The Model Context Protocol (MCP) lets ANY MCP client (Claude
Desktop, IDEs, other agents) call your tools. By wrapping the knowledge graph as
an MCP server, the graph becomes a reusable capability — not glue code locked
inside one script. FastMCP turns annotated Python functions into MCP tools and
resources with almost no boilerplate.

Tools:
  query_graph(cypher)       → run Cypher, return up to 20 rows as JSON
  get_graph_schema()        → labels + relationship types
  find_anomalies(hours=24)  → recent unresolved HIGH/CRITICAL alerts

Resource:
  graph://schema            → the live schema (labels + relationship types)

This is a SERVER. Run it directly (`mcp.run()` speaks MCP over stdio) and wire it
into a client. Claude Desktop config snippet is at the bottom of this file.

Requires:
  - pip install fastmcp

Run (as an MCP server):  python week15/kg_mastery/part3_graphrag/05_fastmcp_server.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import get_driver

try:
    from fastmcp import FastMCP
except ImportError as e:
    print(f"⚠️  Missing dependency: {e.name}")
    print("   pip install fastmcp")
    sys.exit(1)

mcp = FastMCP("Hotel Knowledge Graph MCP Server")


@mcp.tool()
def query_graph(cypher: str) -> str:
    """Run a read-only Cypher query against the hotel knowledge graph.

    Returns up to 20 result rows as a JSON string.
    """
    driver = get_driver()
    try:
        with driver.session() as s:
            rows = [r.data() for r in s.run(cypher)][:20]
        return json.dumps(rows, default=str)
    except Exception as e:  # noqa: BLE001
        return f"ERROR running Cypher: {type(e).__name__}: {e}"
    finally:
        driver.close()


def _schema_text() -> str:
    driver = get_driver()
    try:
        with driver.session() as s:
            labels = [r["label"] for r in s.run("CALL db.labels() YIELD label RETURN label")]
            rels = [
                r["relationshipType"]
                for r in s.run(
                    "CALL db.relationshipTypes() YIELD relationshipType "
                    "RETURN relationshipType"
                )
            ]
        return json.dumps({"labels": labels, "relationship_types": rels})
    except Exception as e:  # noqa: BLE001
        return f"ERROR reading schema: {type(e).__name__}: {e}"
    finally:
        driver.close()


@mcp.tool()
def get_graph_schema() -> str:
    """Return the graph schema: all node labels and relationship types as JSON."""
    return _schema_text()


@mcp.tool()
def find_anomalies(hours: int = 24) -> str:
    """Return unresolved HIGH/CRITICAL alerts raised in the last `hours` hours.

    Returns the device, alert type, severity, message and timestamp as JSON.
    """
    driver = get_driver()
    try:
        with driver.session() as s:
            rows = [
                r.data()
                for r in s.run(
                    """
                    MATCH (d:Device)-[:TRIGGERED]->(a:Alert)
                    WHERE a.resolved = false
                      AND a.severity IN ['HIGH', 'CRITICAL']
                      AND a.ts >= datetime() - duration({hours: $hours})
                    RETURN d.id AS device, a.type AS type, a.severity AS severity,
                           a.message AS message, a.ts AS ts
                    ORDER BY a.ts DESC
                    """,
                    hours=hours,
                )
            ][:20]
        return json.dumps(rows, default=str)
    except Exception as e:  # noqa: BLE001
        return f"ERROR finding anomalies: {type(e).__name__}: {e}"
    finally:
        driver.close()


@mcp.resource("graph://schema")
def schema_resource() -> str:
    """The hotel knowledge-graph schema (labels + relationship types) as JSON."""
    return _schema_text()


if __name__ == "__main__":
    # Servers speak MCP over stdio; there is no check_connection() gate here so
    # the process can start before Neo4j is queried by a client.
    mcp.run()


# ── Claude Desktop config ──────────────────────────────────────────────────
# Add this to ~/Library/Application Support/Claude/claude_desktop_config.json
# (macOS) or %APPDATA%/Claude/claude_desktop_config.json (Windows):
#
# {
#   "mcpServers": {
#     "hotel-knowledge-graph": {
#       "command": "python",
#       "args": [
#         "/Users/altodev5/AgenticCoding/week15/kg_mastery/part3_graphrag/05_fastmcp_server.py"
#       ],
#       "env": {
#         "NEO4J_URI": "bolt://localhost:7687",
#         "NEO4J_USER": "neo4j",
#         "NEO4J_PASSWORD": "mas_memory_2024"
#       }
#     }
#   }
# }
