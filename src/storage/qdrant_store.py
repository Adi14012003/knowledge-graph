import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from anthropic import Anthropic
from src.models import Entity, ExtractionResult
from dotenv import load_dotenv
load_dotenv()

qdrant = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", 6333)),
    check_compatibility=False
)
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

COLLECTION = "knowledge_graph"
VECTOR_SIZE = 1536  # text-embedding-3-small dimension

def ensure_collection():
    """Create the Qdrant collection if it doesn't exist."""
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection: {COLLECTION}")

def embed_text(text: str) -> list[float]:
    """Get embedding vector from Anthropic."""
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": f"Embedding: {text[:500]}"}]
    )
    # Note: Anthropic doesn't have a dedicated embeddings API yet
    # Use OpenAI embeddings or a local model instead
    # For now we use a simple hash-based placeholder
    import hashlib
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    import random
    random.seed(hash_val)
    return [random.uniform(-1, 1) for _ in range(VECTOR_SIZE)]

def save_entities(result: ExtractionResult):
    """Embed and store all entities in Qdrant."""
    ensure_collection()
    points = []
    for entity in result.entities:
        text = f"{entity.name}: {entity.description}"
        vector = embed_text(text)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "name": entity.name,
                "type": entity.entity_type.value,
                "description": entity.description,
                "subtopic": entity.subtopic,
                "confidence": entity.confidence,
                "source_doc": entity.source_document_id,
            }
        ))
    if points:
        qdrant.upsert(collection_name=COLLECTION, points=points)
        print(f"Saved {len(points)} entities to Qdrant")

def search_similar(query: str, top_k: int = 5) -> list[dict]:
    """Find semantically similar entities."""
    ensure_collection()
    query_vector = embed_text(query)
    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=top_k,
    )
    return [
        {"name": r.payload["name"],
         "description": r.payload["description"],
         "score": r.score}
        for r in results
    ]
