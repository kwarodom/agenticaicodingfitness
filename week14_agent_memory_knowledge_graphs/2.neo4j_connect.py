# Install: pip install neo4j langchain-neo4j
from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "mas_memory_2024")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
    print("Connected to Neo4j")
    with driver.session() as session:
        # Create a test node
        session.run(
            "CREATE (a:Agent {name: $name, created: datetime()})",
            name="SensorAgent",
        )
        # Verify
        result = session.run("MATCH (a:Agent) RETURN a.name AS name")
        for r in result:
            print(r["name"])
