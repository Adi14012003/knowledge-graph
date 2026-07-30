import os
from dotenv import load_dotenv
load_dotenv()

def test_neo4j():
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password123"))
    )
    with driver.session() as session:
        result = session.run("RETURN 'Neo4j connected!' AS msg")
        print(result.single()["msg"])
    driver.close()

def test_qdrant():
    from qdrant_client import QdrantClient
    client = QdrantClient(host="localhost", port=6333)
    info = client.get_collections()
    print(f"Qdrant connected! Collections: {info.collections}")

def test_anthropic():
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=20,
        messages=[{"role": "user", "content": "Say: Claude connected!"}]
    )
    print(msg.content[0].text)

if __name__ == "__main__":
    print("Testing connections...\n")
    try:
        test_neo4j()
    except Exception as e:
        print(f"Neo4j FAILED: {e}")
    try:
        test_qdrant()
    except Exception as e:
        print(f"Qdrant FAILED: {e}")
    try:
        test_anthropic()
    except Exception as e:
        print(f"Anthropic FAILED: {e}")
