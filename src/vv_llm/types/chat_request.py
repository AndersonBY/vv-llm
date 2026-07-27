from __future__ import annotations

import warnings
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CapabilityPolicy(str, Enum):
    STRICT = "strict"
    WARN = "warn"
    PASSTHROUGH = "passthrough"


class StructuredOutputCapability(str, Enum):
    NONE = "none"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


class ThinkingCapability(str, Enum):
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    CONFIGURABLE = "configurable"
    ALWAYS_ENABLED = "always_enabled"


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class ThinkingMode(str, Enum):
    DEFAULT = "default"
    ENABLED = "enabled"
    DISABLED = "disabled"
    PROVIDER_DEFINED = "provider_defined"


class ThinkingPreference(BaseModel):
    mode: ThinkingMode = ThinkingMode.DEFAULT
    budget_tokens: int | None = Field(default=None, ge=1)
    value: dict[str, Any] | None = None

    @classmethod
    def default(cls) -> ThinkingPreference:
        return cls(mode=ThinkingMode.DEFAULT)

    @classmethod
    def enabled(cls, budget_tokens: int | None = None) -> ThinkingPreference:
        return cls(mode=ThinkingMode.ENABLED, budget_tokens=budget_tokens)

    @classmethod
    def disabled(cls) -> ThinkingPreference:
        return cls(mode=ThinkingMode.DISABLED)

    @classmethod
    def provider_defined(cls, value: dict[str, Any]) -> ThinkingPreference:
        return cls(mode=ThinkingMode.PROVIDER_DEFINED, value=value)

    def to_provider_value(self) -> dict[str, Any] | None:
        if self.mode is ThinkingMode.DEFAULT:
            return None
        if self.mode is ThinkingMode.DISABLED:
            return {"type": "disabled"}
        if self.mode is ThinkingMode.PROVIDER_DEFINED:
            return dict(self.value or {})

        value: dict[str, Any] = {"type": "enabled"}
        if self.budget_tokens is not None:
            value["budget_tokens"] = self.budget_tokens
        return value


class ModelCapabilities(BaseModel):
    tools: bool = False
    structured_output: StructuredOutputCapability = StructuredOutputCapability.NONE
    input_modalities: set[Modality] = Field(default_factory=lambda: {Modality.TEXT})
    output_modalities: set[Modality] = Field(default_factory=lambda: {Modality.TEXT})
    streaming: bool = True
    parallel_tool_calls: bool = False
    thinking: ThinkingCapability = ThinkingCapability.UNKNOWN

    @classmethod
    def from_legacy(
        cls,
        *,
        function_call_available: bool,
        response_format_available: bool,
        native_multimodal: bool,
    ) -> ModelCapabilities:
        input_modalities = {Modality.TEXT}
        if native_multimodal:
            input_modalities.add(Modality.IMAGE)
        return cls(
            tools=function_call_available,
            structured_output=StructuredOutputCapability.JSON_SCHEMA if response_format_available else StructuredOutputCapability.NONE,
            input_modalities=input_modalities,
        )


class ChatRequestOptions(BaseModel):
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    response_format: Any | None = None
    stream_options: Any | None = None
    audio: Any | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, int] | None = None
    logprobs: bool | None = None
    max_completion_tokens: int | None = None
    metadata: Any | None = None
    modalities: list[Any] | None = None
    n: int | None = None
    parallel_tool_calls: bool | None = None
    prediction: Any | None = None
    presence_penalty: float | None = None
    reasoning_effort: str | None = None
    thinking: ThinkingPreference | dict[str, Any] | None = None
    seed: int | None = None
    service_tier: str | None = None
    stop: str | list[str] | None = None
    store: bool | None = None
    top_logprobs: int | None = None
    user: str | None = None
    timeout: Any | None = None
    provider_options: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def resolved_thinking(self) -> dict[str, Any] | None:
        if isinstance(self.thinking, ThinkingPreference):
            return self.thinking.to_provider_value()
        if self.thinking is None:
            return None
        return dict(self.thinking)


class ChatRequest(BaseModel):
    messages: list[Any]
    model: str | None = None
    stream: bool = False
    options: ChatRequestOptions = Field(default_factory=ChatRequestOptions)
    tools: Any | None = None
    tool_choice: Any | None = None
    skip_cutoff: bool = False
    extra_headers: Any | None = None
    header_context: dict[str, Any] | None = None
    extra_query: Any | None = None
    extra_body: dict[str, Any] | None = None

    def validate_capabilities(
        self,
        capabilities: ModelCapabilities,
        policy: CapabilityPolicy = CapabilityPolicy.WARN,
    ) -> None:
        if policy is CapabilityPolicy.PASSTHROUGH:
            return

        conflicts: list[str] = []
        thinking = self.options.thinking
        if isinstance(thinking, ThinkingPreference) and thinking.mode is not ThinkingMode.DEFAULT:
            if capabilities.thinking is ThinkingCapability.UNSUPPORTED:
                conflicts.append("the model does not support thinking controls")
            elif capabilities.thinking is ThinkingCapability.UNKNOWN:
                conflicts.append("the model's thinking capability is unknown")
            elif capabilities.thinking is ThinkingCapability.ALWAYS_ENABLED and thinking.mode is ThinkingMode.DISABLED:
                conflicts.append("thinking is always enabled for this model")
        if self.tools and not capabilities.tools:
            conflicts.append("the model does not support tools")
        if self.options.response_format is not None and capabilities.structured_output is StructuredOutputCapability.NONE:
            conflicts.append("the model does not support structured output")
        if self.stream and not capabilities.streaming:
            conflicts.append("the model does not support streaming")
        if any(_message_has_image(message) for message in self.messages) and Modality.IMAGE not in capabilities.input_modalities:
            conflicts.append("the model does not support image input")

        if not conflicts:
            return
        message = "; ".join(conflicts)
        if policy is CapabilityPolicy.STRICT:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=2)

    def to_completion_kwargs(self, backend_name: str) -> dict[str, Any]:
        option_values = self.options.model_dump(
            exclude_none=True,
            exclude={"thinking", "provider_options"},
        )
        kwargs: dict[str, Any] = {
            "messages": self.messages,
            "stream": self.stream,
            "skip_cutoff": self.skip_cutoff,
            **option_values,
        }
        if self.model is not None:
            kwargs["model"] = self.model
        if self.tools is not None:
            kwargs["tools"] = self.tools
        if self.tool_choice is not None:
            kwargs["tool_choice"] = self.tool_choice
        if self.extra_headers is not None:
            kwargs["extra_headers"] = self.extra_headers
        if self.header_context is not None:
            kwargs["header_context"] = self.header_context
        if self.extra_query is not None:
            kwargs["extra_query"] = self.extra_query

        thinking = self.options.resolved_thinking()
        if thinking is not None:
            kwargs["thinking"] = thinking

        extra_body = dict(self.extra_body or {})
        extra_body.update(self.options.provider_options.get(backend_name, {}))
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs


def _message_has_image(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(part, dict) and part.get("type") in {"image", "image_url"} for part in content)


__all__ = [
    "CapabilityPolicy",
    "ChatRequest",
    "ChatRequestOptions",
    "Modality",
    "ModelCapabilities",
    "StructuredOutputCapability",
    "ThinkingCapability",
    "ThinkingMode",
    "ThinkingPreference",
]
