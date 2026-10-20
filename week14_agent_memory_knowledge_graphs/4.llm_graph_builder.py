# llm_graph_builder.py — Auto-extract entities from text into Neo4j
# pip install langchain-experimental langchain-neo4j langchain-anthropic python-dotenv
import os
from dotenv import load_dotenv
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document

load_dotenv()

llm = ChatAnthropic(
    model="claude-sonnet-4-6",  # entity extraction benefits from a stronger model
    api_key=os.environ["ANTHROPIC_API_KEY"],
    max_tokens=2048,
)

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="mas_memory_2024",
)

transformer = LLMGraphTransformer(llm=llm)

# Feed any text document into the knowledge graph
docs = [
    Document(
        page_content=(
            "Room 301 HVAC unit failed at 14:23. "
            "Maintenance agent dispatched. "
            "Energy consumption increased by 40%."
        )
    )
]

graph_docs = transformer.convert_to_graph_documents(docs)
graph.add_graph_documents(graph_docs, include_source=True)

# Show what got created
print("Nodes:")
for n in graph_docs[0].nodes:
    print(f"  ({n.id}:{n.type}) {n.properties}")

print("\nRelationships:")
for r in graph_docs[0].relationships:
    print(f"  ({r.source.id})-[:{r.type}]->({r.target.id})")
