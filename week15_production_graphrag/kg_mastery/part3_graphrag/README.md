# Part 3 — Applied GraphRAG with Major Frameworks

Runnable companion scripts for **kg_mastery.pdf Part 3 "Applied GraphRAG with
Major Frameworks."** Each script is **standalone**, imports shared connection +
model config from `../common.py`, and runs against the hotel knowledge graph
loaded in Parts 1–2 (Room/Device/SensorReading/Alert/Staff/MaintenanceJob/Guest/
Supplier). Rooms R101/R203/R305 run hot; alerts AL1/AL2 are unresolved HIGH.

Every script guards its optional framework import and tells you exactly what to
`pip install`, and gates on `check_connection()` first.

## Scripts

| # | Script | Framework | `pip install` | What it shows |
|---|--------|-----------|---------------|---------------|
| 1 | `01_langchain_graphcypher.py` | LangChain | `langchain-neo4j langchain-anthropic` | GraphCypherQAChain: NL question → generated Cypher → answer (+ Neo4jVector Pattern 2 in comments) |
| 2 | `02_langgraph_react_agent.py` | LangGraph | `langgraph langchain-anthropic langchain-core` | ReAct agent with read + write tools; stores findings as `:Event` memory |
| 3 | `03_crewai_crew.py` | CrewAI | `crewai crewai-tools` | Two role-based agents (Analyst → Optimizer) sharing graph-backed tools |
| 4 | `04_google_adk_agent.py` | Google ADK | `google-adk` | Plain-function tools wrapped as `FunctionTool`; Agent + Runner construction demo |
| 5 | `05_fastmcp_server.py` | FastMCP / MCP | `fastmcp` | MCP **server** exposing the graph as tools + a `graph://schema` resource |
| 6 | `06_llamaindex.py` | LlamaIndex | `llama-index llama-index-graph-stores-neo4j llama-index-llms-anthropic` | Build a KG **from documents** (triple extraction) and query it |
| 7 | `07_text2cypher_pipeline.py` | LangChain (hand-built) | `langchain-neo4j langchain-anthropic langchain-core` | Custom Text2Cypher pipeline; the key lesson is **graceful error handling** |
| 8 | `08_agent_memory.py` | neo4j driver (no framework) | *(none extra)* | `AgentMemory` class: episodic (`:Event`) + semantic (`:FACT`) memory layers |

## Prerequisites

- **Neo4j running** with the hotel schema loaded (see `../docker-compose.yml`
  and Parts 1–2). Default creds: `neo4j` / `mas_memory_2024` at
  `bolt://localhost:7687`.
- **`ANTHROPIC_API_KEY`** — needed by scripts 1, 2, 3, 6, 7 (they use
  `claude-sonnet-4-6` via `LLM_MODEL`). Set it in your environment or repo-root
  `.env`.
- **Script 4 (Google ADK)** uses Gemini and needs **Google credentials**
  (`GOOGLE_API_KEY` or Vertex AI ADC) for an actual run. As shipped it is a
  *construction* demo and does **not** call Gemini.
- **Script 5 (FastMCP)** is a **server**, not a one-shot script: run it and wire
  it into an MCP client. The Claude Desktop `claude_desktop_config.json` snippet
  is in a comment at the bottom of that file.

## Run

```bash
# example — script 1
python week15/kg_mastery/part3_graphrag/01_langchain_graphcypher.py
```
