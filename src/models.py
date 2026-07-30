from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime

class SourceType(str, Enum):
    YOUTUBE = "youtube"
    ARXIV = "arxiv"
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"

class Document(BaseModel):
    id: str
    source_url: str
    source_type: SourceType
    title: str
    raw_text: str
    metadata: dict = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    content_hash: str

class EntityType(str, Enum):
    PERSON = "person"
    PAPER = "paper"
    TOOL = "tool"
    MODEL = "model"
    CONCEPT = "concept"
    DATASET = "dataset"
    ORGANIZATION = "organization"
    REPOSITORY = "repository"

class RelationshipType(str, Enum):
    CITES = "CITES"
    BUILT_BY = "BUILT_BY"
    EXPLAINS = "EXPLAINS"
    USED_IN = "USED_IN"
    AUTHORED_BY = "AUTHORED_BY"
    RELATED_TO = "RELATED_TO"

class Relationship(BaseModel):
    target_name: str
    target_type: EntityType
    relationship_type: RelationshipType

class Entity(BaseModel):
    name: str
    entity_type: EntityType
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_span: str
    source_document_id: str
    subtopic: str
    relationships: list[Relationship] = Field(default_factory=list)

class ExtractionResult(BaseModel):
    document_id: str
    is_relevant: bool
    entities: list[Entity] = Field(default_factory=list)
    rejection_reason: Optional[str] = None

class QueryResult(BaseModel):
    answer: str
    sources: list[str]
    gaps_filled: list[str]
    confidence: float
