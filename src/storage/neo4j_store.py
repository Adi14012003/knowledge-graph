import os
from neo4j import GraphDatabase
from src.models import ExtractionResult
from dotenv import load_dotenv
load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password123"))
)

def save_entities(result: ExtractionResult):
    with driver.session() as session:
        for entity in result.entities:
            session.run("""
                MERGE (e:Entity {name: $name})
                SET e.type = $type,
                    e.description = $description,
                    e.confidence = $confidence,
                    e.subtopic = $subtopic,
                    e.source_doc = $source_doc
            """, {
                "name": entity.name,
                "type": entity.entity_type.value,
                "description": entity.description,
                "confidence": entity.confidence,
                "subtopic": entity.subtopic,
                "source_doc": entity.source_document_id,
            })
            for rel in entity.relationships:
                session.run(f"""
                    MERGE (a:Entity {{name: $from_name}})
                    MERGE (b:Entity {{name: $to_name}})
                    MERGE (a)-[r:{rel.relationship_type.value}]->(b)
                """, {
                    "from_name": entity.name,
                    "to_name": rel.target_name,
                })

def search_entities(query: str, limit: int = 10) -> list[dict]:
    words = [w for w in query.lower().split() if len(w) > 3]
    if not words:
        words = query.lower().split()

    with driver.session() as session:
        results = []
        seen = set()

        for word in words:
            rows = session.run("""
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($word)
                   OR toLower(e.description) CONTAINS toLower($word)
                RETURN e.name as name, e.type as type,
                       e.description as description,
                       e.subtopic as subtopic
                LIMIT $limit
            """, {"word": word, "limit": limit})
            for r in rows:
                if r["name"] not in seen:
                    seen.add(r["name"])
                    results.append(dict(r))

        return results[:limit]

def get_related_entities(entity_name: str) -> list[dict]:
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Entity {name: $name})-[]-(related:Entity)
            RETURN DISTINCT related.name as name,
                   related.type as type,
                   related.description as description
            LIMIT 20
        """, {"name": entity_name})
        return [dict(r) for r in result]
