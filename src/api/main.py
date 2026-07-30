import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from src.graph import run_query
from src.extractors.youtube import fetch_youtube
from src.extractors.arxiv import fetch_arxiv
from src.extraction import extract_entities
from src.storage.neo4j_store import save_entities as neo4j_save, search_entities
from src.storage.qdrant_store import save_entities as qdrant_save, search_similar

app = FastAPI(title="Knowledge Graph API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    question: str

class IngestRequest(BaseModel):
    url: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Knowledge Graph API is running", "docs": "/docs"}

@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        return run_query(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
def ingest(req: IngestRequest):
    url = req.url.strip()
    try:
        # Step 1: Fetch content
        if "youtube.com" in url or "youtu.be" in url:
            doc = fetch_youtube(url)
        elif "arxiv.org" in url or url.replace(".", "").isdigit():
            doc = fetch_arxiv(url)
        else:
            raise HTTPException(status_code=400,
                                detail="URL must be YouTube or arXiv")

        # Step 2: Extract entities with Claude
        result = extract_entities(doc)

        # Step 3: Save to Neo4j (graph relationships)
        if result.is_relevant and result.entities:
            neo4j_save(result)
            qdrant_save(result)

        return {
            "document_id": doc.id,
            "title": doc.title,
            "is_relevant": result.is_relevant,
            "entities_found": len(result.entities),
            "saved_to_db": result.is_relevant and len(result.entities) > 0,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/entities")
def list_entities(search: str = ""):
    """Browse what's stored in the knowledge graph."""
    try:
        if search:
            results = search_entities(search)
        else:
            results = search_entities("transformer")
        return {"entities": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/stats")
def graph_stats():
    """How many nodes and relationships are in the graph."""
    from src.storage.neo4j_store import driver
    with driver.session() as session:
        nodes = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
        rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
        types = session.run("MATCH (e:Entity) RETURN DISTINCT e.type as type, count(e) as count ORDER BY count DESC").data()
    return {
        "total_nodes": nodes,
        "total_relationships": rels,
        "breakdown": types
    }


@app.get("/graph/stats")
def graph_stats():
    from src.storage.neo4j_store import driver
    with driver.session() as session:
        nodes = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
        rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
        types = session.run("""
            MATCH (e:Entity)
            RETURN DISTINCT e.type as type, count(e) as count
            ORDER BY count DESC
        """).data()
    return {
        "total_nodes": nodes,
        "total_relationships": rels,
        "breakdown": types
    }
