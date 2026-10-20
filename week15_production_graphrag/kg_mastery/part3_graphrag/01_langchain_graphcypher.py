#!/usr/bin/env python3
"""Part 3.1 — LangChain GraphCypherQAChain (kg_mastery.pdf §3.1).

KEY INSIGHT: The fastest path to GraphRAG. GraphCypherQAChain wires an LLM to a
Neo4j graph: it (1) reads your live schema, (2) translates a natural-language
question into Cypher, (3) runs that Cypher, then (4) turns the rows back into a
plain-English answer. You write zero query code — but you MUST opt in with
`allow_dangerous_requests=True` because the LLM generates executable Cypher.

  Pattern 1 (this script): structured Q&A over the graph via generated Cypher.
  Pattern 2 (commented below): Neo4jVector semantic retrieval (needs Part 2.6
  embeddings).

Demo questions:
  - "Which rooms had temperature above 28C today?"
  - "Which devices triggered HIGH alerts?"

Requires:
  - ANTHROPIC_API_KEY in your environment / repo-root .env
  - pip install langchain-neo4j langchain-anthropic

Run:  python week15/kg_mastery/part3_graphrag/01_langchain_graphcypher.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, LLM_MODEL, check_connection


def build_chain():
    # ── Neo4jGraph: prefer the maintained package, fall back to community ──
    try:
        from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
    except ImportError:
        try:
            from langchain_community.graphs import Neo4jGraph
            from langchain.chains import GraphCypherQAChain
        except ImportError:
            print("⚠️  Missing dependency: langchain-neo4j")
            print("   pip install langchain-neo4j langchain-anthropic")
            sys.exit(1)

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        print("⚠️  Missing dependency: langchain-anthropic")
        print("   pip install langchain-neo4j langchain-anthropic")
        sys.exit(1)

    graph = Neo4jGraph(
        url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD
    )
    graph.refresh_schema()  # introspect labels/rels/props the LLM will see
    print("── Graph schema the LLM will use ──")
    print(graph.schema)
    print("───────────────────────────────────\n")

    llm = ChatAnthropic(model=LLM_MODEL, temperature=0)

    chain = GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        verbose=True,
        return_intermediate_steps=True,
        allow_dangerous_requests=True,  # required: LLM emits executable Cypher
    )
    return chain


def ask(chain, question):
    print(f"\n=== Q: {question} ===")
    result = chain.invoke({"query": question})

    # The generated Cypher lives in the intermediate steps.
    for step in result.get("intermediate_steps", []):
        if "query" in step:
            print("Generated Cypher:")
            print(f"  {step['query']}")
    print("Answer:")
    print(f"  {result['result']}")
    return result


def main():
    chain = build_chain()
    ask(chain, "Which rooms had temperature above 28C today?")
    ask(chain, "Which devices triggered HIGH alerts?")

    # ── Pattern 2: Neo4jVector semantic retrieval (requires Part 2.6) ──────
    # Once room embeddings exist (see part2_building/05_vector_embeddings.py),
    # you can retrieve by meaning instead of by generated Cypher:
    #
    #   from langchain_neo4j import Neo4jVector
    #   from langchain_openai import OpenAIEmbeddings
    #
    #   store = Neo4jVector.from_existing_index(
    #       OpenAIEmbeddings(model="text-embedding-3-small"),
    #       url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD,
    #       index_name="room_embeddings",
    #       node_label="Room",
    #       text_node_property="description",
    #       embedding_node_property="embedding",
    #   )
    #   docs = store.similarity_search("quiet room with ocean view", k=3)
    #   for d in docs:
    #       print(d.page_content)


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set.")
        print("   export ANTHROPIC_API_KEY=sk-ant-...  (or add it to repo-root .env)")
        sys.exit(1)
    if not check_connection():
        sys.exit(1)
    main()
