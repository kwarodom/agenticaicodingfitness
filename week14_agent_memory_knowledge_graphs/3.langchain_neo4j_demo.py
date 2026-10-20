# pip install langchain langchain-neo4j langchain-anthropic python-dotenv
import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_anthropic import ChatAnthropic

load_dotenv()

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="mas_memory_2024",
)

# Refresh schema (auto-detects nodes/relationships)
graph.refresh_schema()
print(graph.schema)

# Natural language → Cypher → Answer (Claude)
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",  # fast & cheap; swap to claude-sonnet-4-6 for higher quality
    api_key=os.environ["ANTHROPIC_API_KEY"],
    max_tokens=1024,
)

chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
)

answer = chain.invoke({"query": "Which agents exist in the graph?"})
print(answer["result"])
