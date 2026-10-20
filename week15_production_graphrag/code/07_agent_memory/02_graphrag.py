#!/usr/bin/env python3
"""Lesson 07.2 — GraphRAG: ask the graph in plain English.

Prereq: run 01_memory_demo.py first (so there's data), with Neo4j running.
Needs:  pip install langchain-neo4j langchain-anthropic  (in requirements.txt)

Run:  python week15/code/07_agent_memory/02_graphrag.py

You don't have to write Cypher by hand. GraphRAG = an LLM translates your
natural-language question into Cypher, runs it, and phrases the answer. This is
RAG (folder 05) but retrieving over a GRAPH instead of a pile of text.

Set verbose=True below to watch the Cypher the LLM generates.
"""
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from agent_memory import URI, USER, PASSWORD


def main():
    from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
    from langchain_anthropic import ChatAnthropic

    graph = Neo4jGraph(url=URI, username=USER, password=PASSWORD)
    graph.refresh_schema()  # show the LLM what's in the graph

    llm = ChatAnthropic(model="claude-haiku-4-5-20251001",
                        api_key=os.environ["ANTHROPIC_API_KEY"], max_tokens=512)
    chain = GraphCypherQAChain.from_llm(
        llm, graph=graph, verbose=True, allow_dangerous_requests=True,
    )

    for question in [
        "What faults were detected?",
        "What happened in Room 301?",
        "Which equipment was involved in a work order?",
    ]:
        print(f"\nQ: {question}")
        print(f"A: {chain.invoke({'query': question})['result']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print("⚠️  Could not run GraphRAG.")
        print(f"   ({type(e).__name__}: {e})")
        print("   Make sure: (1) Neo4j is running, (2) you ran 01_memory_demo.py first,")
        print("   (3) pip install langchain-neo4j langchain-anthropic")
