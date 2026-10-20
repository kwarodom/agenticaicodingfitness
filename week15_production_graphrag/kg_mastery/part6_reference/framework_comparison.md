# Part 6 — Framework & Architecture Reference

`kg_mastery.pdf` Parts 3 & 6. Three reference tables for choosing how to build a
GraphRAG / agent-over-knowledge-graph system.

## 1. Agent / GraphRAG framework comparison

| Framework | Best for | Neo4j integration | LLM support |
|-----------|----------|-------------------|-------------|
| **LangChain** | Quick chains, NL-to-Cypher, RAG prototypes | First-class — `Neo4jGraph`, `GraphCypherQAChain`, `Neo4jVector` | Any (Anthropic, OpenAI, local, …) |
| **LangGraph** | Stateful, multi-step ReAct agents with loops & control flow | Via LangChain tools / custom `@tool` wrapping the driver | Any (LangChain models) |
| **CrewAI** | Multi-agent "crews" with roles & delegation | Custom tools around the Neo4j driver | Any (LangChain models) |
| **Google ADK** | Google-ecosystem agents (Gemini, Vertex, GCP) | Custom tools / function calling | Gemini-centric |
| **FastMCP** | Exposing the graph as MCP tools to ANY MCP client | Custom MCP server wrapping the driver | Model-agnostic (MCP clients) |
| **LlamaIndex** | Document-heavy RAG, property-graph index, ingestion | `Neo4jPropertyGraphStore`, `Neo4jVectorStore` | Any (OpenAI, Anthropic, local) |

## 2. GraphRAG architecture patterns

| Pattern | How it works | Best for | Complexity |
|---------|--------------|----------|------------|
| **NL-to-Cypher** | LLM translates a question into a Cypher query, runs it, summarizes the rows | Structured Q&A over a known schema; analytics-style questions | Low |
| **Vector + Graph Expand** | Vector-search seed nodes/chunks, then traverse relationships to pull in connected context before answering | Document RAG that needs surrounding context; "find similar, then explain how they connect" | Medium |
| **Agent with KG Tools** | An agent picks from hand-written graph tools (each wrapping fixed Cypher), reasons over results in a loop, and can write back | Multi-step workflows, actions/side-effects, reliability-critical apps | High |

## 3. Tool-selection decision guide

- **If you want the fastest prototype of question → answer over a known schema → use LangChain `GraphCypherQAChain` (NL-to-Cypher).**
- **If you want multi-step reasoning, loops, or write-back to the graph → use LangGraph `create_react_agent` with custom `@tool`s (Agent with KG Tools).**
- **If you want several specialized agents that delegate to each other → use CrewAI.**
- **If you're all-in on the Google/Gemini stack → use Google ADK.**
- **If you want the graph usable by many AI clients / IDEs as standard tools → use FastMCP to expose it over MCP.**
- **If your problem is document-heavy ingestion + retrieval → use LlamaIndex's property-graph / vector stores.**
- **If answers need both semantic similarity AND structural context → use the Vector + Graph Expand pattern (combine `Neo4jVector` retrieval with a traversal step).**
- **If you need maximum safety/reliability → don't let the LLM write raw Cypher; wrap fixed queries in named tools (Agent with KG Tools) instead of NL-to-Cypher.**

> Rule of thumb (PDF): start at the lowest complexity that answers the question.
> Reach for NL-to-Cypher first, add Vector+Graph Expand when context matters, and
> graduate to a tool-using agent only when you need multi-step reasoning or writes.
