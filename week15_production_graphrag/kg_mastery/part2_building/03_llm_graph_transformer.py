#!/usr/bin/env python3
"""Part 2.2 — Unstructured text → graph with LLMGraphTransformer (kg_mastery.pdf §2.2).

Takes free-text maintenance reports and lets an LLM extract a property graph:
nodes (Room, Device, Alert, Staff, MaintenanceJob) and the relationships between
them (HAS_DEVICE, TRIGGERED, ASSIGNED_TO, RESOLVED_BY). LangChain's
`LLMGraphTransformer` does the extraction; `Neo4jGraph.add_graph_documents`
writes it.

NOTE: these LLM-extracted nodes are SEPARATE from the curated hotel dataset
loaded in Part 1. Their ids/labels come from the model's reading of the text
(e.g. "Room 301", device "C-7"), so don't expect them to MERGE onto the R1xx
graph. With include_source=True each extracted node also links back to its
source :Document, which is handy for provenance.

Requires:
  - ANTHROPIC_API_KEY in your environment / repo-root .env
  - pip install langchain-experimental langchain-neo4j langchain-anthropic langchain-core

Run:  python week15/kg_mastery/part2_building/03_llm_graph_transformer.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import common.py
from common import check_connection, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, LLM_MODEL

REPORTS_PATH = Path(__file__).resolve().parent / "sample_data" / "maintenance_reports.txt"


def main():
    # --- guard optional imports with an actionable message ---
    try:
        from langchain_experimental.graph_transformers import LLMGraphTransformer
        from langchain_neo4j import Neo4jGraph
        from langchain_anthropic import ChatAnthropic
        from langchain_core.documents import Document
    except ImportError as e:
        print(f"⚠️  Missing dependency: {e.name}")
        print("   pip install langchain-experimental langchain-neo4j "
              "langchain-anthropic langchain-core")
        sys.exit(1)

    if not REPORTS_PATH.exists():
        print(f"⚠️  Missing reports file: {REPORTS_PATH}")
        sys.exit(1)

    # --- read paragraphs, one Document each ---
    raw = REPORTS_PATH.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    docs = [Document(page_content=p, metadata={"source": REPORTS_PATH.name, "para": i})
            for i, p in enumerate(paragraphs)]
    print(f"Loaded {len(docs)} maintenance-report paragraphs.")

    # --- build the LLM + transformer (faithful to the PDF) ---
    llm = ChatAnthropic(model=LLM_MODEL, temperature=0)
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["Room", "Device", "Alert", "Staff", "MaintenanceJob"],
        allowed_relationships=["HAS_DEVICE", "TRIGGERED", "ASSIGNED_TO", "RESOLVED_BY"],
        node_properties=["name", "type", "status"],
    )

    print("Extracting graph documents with the LLM (this calls the API)...")
    graph_docs = transformer.convert_to_graph_documents(docs)

    n_nodes = sum(len(gd.nodes) for gd in graph_docs)
    n_rels = sum(len(gd.relationships) for gd in graph_docs)
    print(f"Extracted {n_nodes} nodes and {n_rels} relationships.")

    # --- write to Neo4j, keeping a link to the source Document ---
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
    graph.add_graph_documents(graph_docs, include_source=True)
    print("✅ Written to Neo4j (with :Document source nodes via include_source=True).")
    print("   These are separate from the curated R1xx hotel dataset.")


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set.")
        print("   export ANTHROPIC_API_KEY=sk-ant-...  (or add it to repo-root .env)")
        sys.exit(1)
    if not check_connection():
        sys.exit(1)
    main()
