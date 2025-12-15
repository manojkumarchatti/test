from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    source: str
    page: Optional[int] = None
    title: Optional[str] = None


class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: DocumentMetadata


class IngestResponse(BaseModel):
    document_id: str
    chunks: int


class QuestionRequest(BaseModel):
    question: str
    top_k: int = Field(6, ge=1, le=20)
    mmr_k: int = Field(4, ge=1, le=20)
    use_reranker: bool = True


class SourceAttribution(BaseModel):
    snippet: str
    metadata: DocumentMetadata
    score: float


class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceAttribution]
    from_cache: bool = False
