from __future__ import annotations

from pydantic import BaseModel, Field


class EmbeddingData(BaseModel):
    index: int
    embedding: list[float]
    object: str = "embedding"
    text: str | None = None
    metadata: dict | None = None


class EmbeddingUsage(BaseModel):
    prompt_tokens: int | None = None
    total_tokens: int | None = None


class EmbeddingResponse(BaseModel):
    model: str
    data: list[EmbeddingData] = Field(default_factory=list)
    usage: EmbeddingUsage | None = None
    raw_response: dict | None = None


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: str | dict | None = None
    id: str | None = None
    metadata: dict | None = None


class RerankUsage(BaseModel):
    search_units: int | None = None
    total_tokens: int | None = None


class RerankResponse(BaseModel):
    model: str
    results: list[RerankResult] = Field(default_factory=list)
    usage: RerankUsage | None = None
    raw_response: dict | None = None


__all__ = [
    "EmbeddingData",
    "EmbeddingUsage",
    "EmbeddingResponse",
    "RerankResult",
    "RerankUsage",
    "RerankResponse",
]
