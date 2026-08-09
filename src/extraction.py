import os
import anthropic
from src.models import Document, Entity, EntityType, RelationshipType, Relationship, ExtractionResult
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def clean(text: str) -> str:
    if not text:
        return ""
    return text.replace('\u2028', ' ').replace('\u2029', ' ').replace('\u0000', '')

EXTRACTION_TOOL = {
    "name": "extract_knowledge",
    "description": "Extract structured entities from AI/ML content",
    "input_schema": {
        "type": "object",
        "properties": {
            "is_relevant": {"type": "boolean"},
            "rejection_reason": {"type": "string"},
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "entity_type": {"type": "string", "enum": [e.value for e in EntityType]},
                        "description": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "source_span": {"type": "string"},
                        "subtopic": {"type": "string", "enum": [
                            "Foundations", "Deep Learning", "LLMs & NLP",
                            "Computer Vision", "MLOps", "AI System Design",
                            "Tools & Frameworks", "GitHub Resources", "Research", "Career"
                        ]},
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "target_name": {"type": "string"},
                                    "target_type": {"type": "string", "enum": [e.value for e in EntityType]},
                                    "relationship_type": {"type": "string", "enum": [r.value for r in RelationshipType]}
                                },
                                "required": ["target_name", "target_type", "relationship_type"]
                            }
                        }
                    },
                    "required": ["name", "entity_type", "description", "confidence", "source_span", "subtopic"]
                }
            }
        },
        "required": ["is_relevant", "entities"]
    }
}

SYSTEM_PROMPT = """You are a knowledge extraction specialist for AI Engineering.
CRITICAL: Always call extract_knowledge with is_relevant (true/false) and entities (array).
Extract people, tools, models, papers, concepts from AI/ML content.
If content is relevant to AI Engineering, set is_relevant=true and extract entities."""

def extract_entities(doc: Document) -> ExtractionResult:
    text = clean(doc.raw_text[:8000])
    title = clean(doc.title)
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            tools=[EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": "extract_knowledge"},
            messages=[{
                "role": "user",
                "content": f"Extract knowledge from:\nTitle: {title}\nSource: {doc.source_url}\nType: {doc.source_type.value}\n\nContent:\n{text}"
            }]
        )
        tool_block = next((b for b in response.content if b.type == "tool_use"), None)
        if not tool_block:
            return ExtractionResult(document_id=doc.id, is_relevant=False, entities=[], rejection_reason="No tool call")
        raw = tool_block.input
        entities = []
        for e in raw.get("entities", []):
            try:
                relationships = [Relationship(**r) for r in e.get("relationships", [])]
                entity = Entity(
                    name=clean(e["name"]),
                    entity_type=EntityType(e["entity_type"]),
                    description=clean(e["description"]),
                    confidence=float(e["confidence"]),
                    source_span=clean(e["source_span"]),
                    source_document_id=doc.id,
                    subtopic=e["subtopic"],
                    relationships=relationships,
                )
                entities.append(entity)
            except Exception as ex:
                print(f"Skipping entity: {ex}")
        return ExtractionResult(
            document_id=doc.id,
            is_relevant=raw.get("is_relevant", False),
            entities=entities,
            rejection_reason=raw.get("rejection_reason"),
        )
    except Exception as e:
        print(f"Extraction failed: {e}")
        return ExtractionResult(document_id=doc.id, is_relevant=False, entities=[], rejection_reason=str(e))
