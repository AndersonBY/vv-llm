from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field


ResponseType = TypeVar("ResponseType")
FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "refusal", "other"]


class ResponseMetadata(BaseModel):
    provider: str | None = None
    model: str | None = None
    response_id: str | None = None
    request_id: str | None = None
    finish_reason: str | None = Field(default=None, description="Unmodified provider finish/stop reason, or Responses status/reason.")
    finish_reason_normalized: FinishReason | None = None
    refusal: str | None = None
    stream_terminated: bool | None = Field(default=None, description="Whether an upstream termination signal has been observed; not a success verdict.")
    stream_exhausted: bool | None = Field(default=None, description="True only after normal iterator exhaustion, never on early close or error.")
    termination_signals: list[str] | None = None
    attempts: int = Field(default=1, ge=1)
    latency_ms: float | None = Field(default=None, ge=0)
    fallback_index: int = Field(default=0, ge=0)
    attributes: dict[str, Any] = Field(default_factory=dict)


def response_metadata(response: Any, **execution: Any) -> ResponseMetadata:
    metadata = getattr(response, "metadata", None)
    if not isinstance(metadata, ResponseMetadata):
        metadata = ResponseMetadata(finish_reason=getattr(response, "finish_reason", None))
    for name, value in execution.items():
        if name == "attributes":
            metadata.attributes.update(value)
        else:
            setattr(metadata, name, value)
    return metadata


@dataclass(frozen=True)
class CompletionResult(Generic[ResponseType]):
    response: ResponseType
    metadata: ResponseMetadata


__all__ = ["CompletionResult", "FinishReason", "ResponseMetadata"]
