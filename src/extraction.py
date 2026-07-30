import os
import anthropic
from src.models import Document, Entity, EntityType, RelationshipType, Relationship, ExtractionResult
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

EXTRACTION_TOOL = {
    "name": "extract_knowledge",
    "description": "Extract structured entities from AI/ML content",
    "input_schema": {
        "type": "object",
        "properties": {
            "is_relevant": {
                "type": "boolean",
                "description": "Is this content relevant to AI Engineering? Must always be set to true or false."
            },
            "rejection_reason": {
                "type": "string",
                "description": "If not relevant, explain why. Leave empty string if relevant."
            },
            "entities": {
                "type": "array",
                "description": "List of extracted entities. Empty array if not relevant.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "entity_type": {
                            "type": "string",
                            "enum": [e.value for e in EntityType]
                        },
                        "description": {"type": "string"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "source_span": {"type": "string"},
                        "subtopic": {
                            "type": "string",
                            "enum": [
                                "Foundations", "Deep Learning", "LLMs & NLP",
                                "Computer Vision", "MLOps", "AI System Design",
                                "Tools & Frameworks", "GitHub Resources",
                                "Research", "Career"
                            ]
                        },
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "target_name": {"type": "string"},
                                    "target_type": {
                                        "type": "string",
                                        "enum": [e.value for e in EntityType]
                                    },
                                    "relationship_type": {
                                        "type": "string",
                                        "enum": [r.value for r in RelationshipType]
                                    }
                                },
                                "required": ["target_name", "target_type", "relationship_type"]
                            }
                        }
                    },
                    "required": ["name", "entity_type", "description",
                                 "confidence", "source_span", "subtopic"]
                }
            }
        },
        "required": ["is_relevant", "entities"]
    }
}

SYSTEM_PROMPT = """You are a knowledge extraction specialist for AI Engineering.

CRITICAL: You MUST always call the extract_knowledge tool with ALL required fields.
- is_relevant: always set to true or false (never omit this)
- entities: always set to an array (empty array [] if not relevant)

Extract people, tools, models, papers, concepts, datasets, organizations from the content.
Only extract entities relevant to AI/ML engineering."""

def extract_entities(doc: Document) -> ExtractionResult:
    # Truncate to avoid token limits — take first 8000 chars for YouTube
    # (transcripts are very long)
    text = doc.raw_text[:8000]

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            tools=[EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": "extract_knowledge"},
            messages=[{
                "role": "user",
                "content": f"Extract knowledge entities from this content.\n\nTitle: {doc.title}\nSource: {doc.source_url}\nType: {doc.source_type.value}\n\nContent:\n{text}"
            }]
        )

        # Find the tool_use block
        tool_block = None
        for block in response.content:
            if block.type == "tool_use":
                tool_block = block
                break

        if not tool_block:
            return ExtractionResult(
                document_id=doc.id,
                is_relevant=False,
                entities=[],
                rejection_reason="Claude did not call the extraction tool"
            )

        raw = tool_block.input

        # Safe defaults if fields are missing
        is_relevant = raw.get("is_relevant", False)
        rejection_reason = raw.get("rejection_reason", "")

        entities = []
        for e in raw.get("entities", []):
            try:
                relationships = [
                    Relationship(**r) for r in e.get("relationships", [])
                ]
                entity = Entity(
                    name=e["name"],
                    entity_type=EntityType(e["entity_type"]),
                    description=e["description"],
                    confidence=float(e["confidence"]),
                    source_span=e["source_span"],
                    source_document_id=doc.id,
                    subtopic=e["subtopic"],
                    relationships=relationships,
                )
                entities.append(entity)
            except Exception as entity_error:
                print(f"Skipping malformed entity: {entity_error}")
                continue

        return ExtractionResult(
            document_id=doc.id,
            is_relevant=is_relevant,
            entities=entities,
            rejection_reason=rejection_reason if rejection_reason else None,
        )

    except Exception as e:
        print(f"Extraction failed for {doc.id}: {e}")
        return ExtractionResult(
            document_id=doc.id,
            is_relevant=False,
            entities=[],
            rejection_reason=f"Extraction error: {str(e)}"
        )
