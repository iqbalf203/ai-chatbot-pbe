from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class KnowledgeDocumentChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class KnowledgeIngestionResult(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    status: str = "completed"
