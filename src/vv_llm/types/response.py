from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field


ResponseType = TypeVar("ResponseType")


class ResponseMetadata(BaseModel):
    provider: str | None = None
    model: str | None = None
    response_id: str | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    attempts: int = Field(default=1, ge=1)
    latency_ms: float | None = Field(default=None, ge=0)
    fallback_index: int = Field(default=0, ge=0)
    attributes: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class CompletionResult(Generic[ResponseType]):
    response: ResponseType
    metadata: ResponseMetadata


__all__ = ["CompletionResult", "ResponseMetadata"]
