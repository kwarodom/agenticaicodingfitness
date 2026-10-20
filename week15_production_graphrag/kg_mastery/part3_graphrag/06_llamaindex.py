#!/usr/bin/env python3
"""Part 3.6 — LlamaIndex KnowledgeGraphIndex (kg_mastery.pdf §3.6).

KEY INSIGHT: LlamaIndex can BUILD a knowledge graph FROM documents: it extracts
(subject, predicate, object) triples with the LLM and stores them in Neo4j via
Neo4jGraphStore. You then query in natural language and it retrieves by walking
the extracted graph (keyword retriever_mode) instead of pure vector similarity.
This is the document-to-graph-to-answer pipeline in one library.

Pipeline:
  Settings.llm = Anthropic(model=LLM_MODEL)
  Neo4jGraphStore  → StorageContext
  KnowledgeGraphIndex.from_documents(docs)   # extracts + stores triples
  index.as_query_engine(retriever_mode='keyword')

Source docs: ./sample_docs/ (created here) — a short hotel HVAC incident log.

Requires:
  - ANTHROPIC_API_KEY in your environment / repo-root .env
  - pip install llama-index llama-index-graph-stores-neo4j llama-index-llms-anthropic

Run:  python week15/kg_mastery/part3_graphrag/06_llamaindex.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import (
    NEO4J_USER, NEO4J_PASSWORD, NEO4J_URI, LLM_MODEL, check_connection,
)

try:
    from llama_index.core import (
        KnowledgeGraphIndex, StorageContext, Settings, SimpleDirectoryReader,
    )
    from llama_index.graph_stores.neo4j import Neo4jGraphStore
    from llama_index.llms.anthropic import Anthropic
except ImportError as e:
    print(f"⚠️  Missing dependency: {e.name}")
    print("   pip install llama-index llama-index-graph-stores-neo4j "
          "llama-index-llms-anthropic")
    sys.exit(1)

DOCS_DIR = Path(__file__).resolve().parent / "sample_docs"


def build_index():
    Settings.llm = Anthropic(model=LLM_MODEL)

    graph_store = Neo4jGraphStore(
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        url=NEO4J_URI,
    )
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    print(f"Loading documents from {DOCS_DIR} ...")
    documents = SimpleDirectoryReader(str(DOCS_DIR)).load_data()
    print(f"Loaded {len(documents)} document(s). Extracting triples with the LLM...")

    index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=5,
        include_embeddings=False,
    )
    print("✅ KnowledgeGraphIndex built and triples stored in Neo4j.")
    return index


def main():
    index = build_index()
    query_engine = index.as_query_engine(
        retriever_mode="keyword",
        response_mode="tree_summarize",
    )
    question = "What is wrong with room R305 and what action is recommended?"
    print(f"\n=== Q: {question} ===")
    response = query_engine.query(question)
    print("Answer:")
    print(response)


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set.")
        print("   export ANTHROPIC_API_KEY=sk-ant-...  (or add it to repo-root .env)")
        sys.exit(1)
    if not check_connection():
        sys.exit(1)
    main()
