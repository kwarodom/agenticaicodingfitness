#!/usr/bin/env python3
"""Part 3.7 — Custom Text2Cypher pipeline (kg_mastery.pdf §3.7).

KEY INSIGHT: Frameworks are convenient, but a hand-built pipeline gives you full
control — and the single most important production lesson is ERROR HANDLING. An
LLM WILL eventually emit invalid Cypher. Wrap graph.query in try/except so a bad
query degrades gracefully (log it, return a polite fallback) instead of crashing
the whole agent.

Two-stage LCEL pipeline:
  1. cypher_prompt → llm → StrOutputParser   (schema injected; rules: always
     LIMIT 20, prefer OPTIONAL MATCH, return ONLY Cypher)
  2. run the Cypher  (wrapped in try/except — the key insight)
  3. answer_prompt → llm → StrOutputParser   (rows → natural-language answer)

Demo: "How many rooms had energy consumption above 5 kWh yesterday?"

Requires:
  - ANTHROPIC_API_KEY in your environment / repo-root .env
  - pip install langchain-neo4j langchain-anthropic langchain-core

Run:  python week15/kg_mastery/part3_graphrag/07_text2cypher_pipeline.py
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, LLM_MODEL, check_connection

try:
    from langchain_anthropic import ChatAnthropic
    from langchain_neo4j import Neo4jGraph
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError as e:
    print(f"⚠️  Missing dependency: {e.name}")
    print("   pip install langchain-neo4j langchain-anthropic langchain-core")
    sys.exit(1)


CYPHER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert Neo4j engineer. Convert the user's question into ONE "
     "Cypher query for the hotel knowledge graph.\n"
     "Schema:\n{schema}\n\n"
     "Rules:\n"
     "- Always add LIMIT 20.\n"
     "- Prefer OPTIONAL MATCH so missing data does not drop rows.\n"
     "- Return ONLY the Cypher query. No prose, no markdown fences."),
    ("human", "{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer hotel-operations questions. Given the user's question and the "
     "raw query result, write a concise natural-language answer. If the result "
     "is empty or an error, say so plainly and do not invent data."),
    ("human", "Question: {question}\n\nQuery result:\n{data}"),
])


def _strip_fences(text: str) -> str:
    """Remove ```...``` markdown fences the LLM sometimes adds."""
    text = re.sub(r"^```(?:cypher)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def text2cypher_query(question: str, graph, llm) -> str:
    # Stage 1: generate Cypher.
    cypher_chain = CYPHER_PROMPT | llm | StrOutputParser()
    cypher = _strip_fences(
        cypher_chain.invoke({"schema": graph.schema, "question": question})
    )
    print("Generated Cypher:")
    print(f"  {cypher}")

    # Stage 2: run it — THE KEY INSIGHT: never let bad Cypher crash the pipeline.
    try:
        data = graph.query(cypher)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  Cypher execution failed: {type(e).__name__}: {e}")
        data = f"QUERY_ERROR: {e}"  # graceful fallback passed to the answerer

    # Stage 3: turn rows (or the error) into a natural-language answer.
    answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
    answer = answer_chain.invoke({"question": question, "data": str(data)})
    return answer


def main():
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
    graph.refresh_schema()
    llm = ChatAnthropic(model=LLM_MODEL, temperature=0)

    question = "How many rooms had energy consumption above 5 kWh yesterday?"
    print(f"=== Q: {question} ===")
    answer = text2cypher_query(question, graph, llm)
    print("Answer:")
    print(f"  {answer}")


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set.")
        print("   export ANTHROPIC_API_KEY=sk-ant-...  (or add it to repo-root .env)")
        sys.exit(1)
    if not check_connection():
        sys.exit(1)
    main()
