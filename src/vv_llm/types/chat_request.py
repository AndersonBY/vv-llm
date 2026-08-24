from __future__ import annotations

import warnings
from collections.abc import Mapping
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr


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
    _contract_extensions: dict[str, Any] = PrivateAttr(default_factory=dict)
    _contract_passthrough_options: dict[str, Any] = PrivateAttr(default_factory=dict)

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
    thinking: dict[str, Any] | ThinkingPreference | None = None
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
    _contract_extensions: dict[str, Any] = PrivateAttr(default_factory=dict)

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

    @classmethod
    def from_contract(cls, value: Mapping[str, Any]) -> ChatRequest:
        """Decode a canonical contract request into the runtime representation.

        The canonical contract keeps ``stream`` inside ``options`` and requires
        an explicit model. Runtime-only transport controls are intentionally not
        accepted here; callers can add them after decoding when making a live
        request. Provider-neutral ``x_*`` extensions and schema fields that have
        no runtime transport equivalent are retained for a later round trip.
        """
        if not isinstance(value, Mapping):
            raise TypeError("canonical chat request must be a mapping")

        unknown = _unknown_contract_keys(value, _CONTRACT_REQUEST_FIELDS)
        if unknown:
            raise ValueError(f"unsupported canonical chat request field(s): {', '.join(unknown)}")

        model = value.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("canonical chat request requires a non-empty model")
        messages = value.get("messages")
        if not isinstance(messages, list):
            raise TypeError("canonical chat request messages must be a list")

        raw_options = value.get("options", {})
        if not isinstance(raw_options, Mapping):
            raise TypeError("canonical chat request options must be an object")
        option_values = dict(raw_options)
        stream = option_values.pop("stream", False)
        if not isinstance(stream, bool):
            raise TypeError("canonical chat request options.stream must be a boolean")

        option_extensions = {key: option_values.pop(key) for key in tuple(option_values) if isinstance(key, str) and key.startswith("x_")}
        canonical_option_fields = set(ChatRequestOptions.model_fields) - {"timeout", "provider_options"}
        passthrough_options = {key: option_values.pop(key) for key in tuple(option_values) if key == "max_tokens_details"}
        unsupported_options = _unknown_contract_keys(option_values, canonical_option_fields)
        if unsupported_options:
            raise ValueError(f"unsupported canonical chat option(s): {', '.join(unsupported_options)}")

        options = ChatRequestOptions(**option_values)
        options._contract_extensions = option_extensions
        options._contract_passthrough_options = passthrough_options

        tools = value.get("tools")
        if tools is not None:
            if not isinstance(tools, list):
                raise TypeError("canonical chat request tools must be a list")
            tools = [_tool_from_contract(tool) for tool in tools]

        extra_body = value.get("extra_body")
        if extra_body is not None and not isinstance(extra_body, Mapping):
            raise TypeError("canonical chat request extra_body must be an object")

        request = cls(
            model=model,
            messages=[_message_from_contract(message) for message in messages],
            stream=stream,
            options=options,
            tools=tools,
            tool_choice=value.get("tool_choice"),
            extra_body=dict(extra_body) if isinstance(extra_body, Mapping) else None,
        )
        request._contract_extensions = {key: value[key] for key in value if isinstance(key, str) and key.startswith("x_")}
        return request

    def to_contract(self) -> dict[str, Any]:
        """Encode semantic request fields into canonical contract JSON.

        ``extra_headers``, ``header_context``, ``extra_query``, ``skip_cutoff``,
        ``timeout`` and backend-specific ``provider_options`` are runtime
        transport controls and are deliberately excluded from canonical JSON.
        """
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("canonical chat request requires a non-empty model")

        options = self.options.model_dump(
            exclude_none=True,
            exclude={"thinking", "provider_options", "timeout"},
        )
        if self.options.thinking is not None:
            options["thinking"] = _thinking_to_contract(self.options.thinking)
        options.update(self.options._contract_passthrough_options)
        options.update(self.options._contract_extensions)
        options["stream"] = self.stream

        result: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_contract(message) for message in self.messages],
            "options": options,
        }
        if self.tools is not None:
            result["tools"] = [_tool_to_contract(tool) for tool in self.tools]
        if self.tool_choice is not None:
            result["tool_choice"] = self.tool_choice
        if self.extra_body is not None:
            result["extra_body"] = dict(self.extra_body)
        result.update(self._contract_extensions)
        return result

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
        # ``max_tokens_details`` is part of the canonical request contract but
        # is not currently accepted by the OpenAI SDK's typed method
        # signature.  ``extra_body`` is merged into the actual JSON request
        # body by the SDK, so retaining it here keeps the field on the wire
        # without adding an SDK-version-specific keyword argument.
        extra_body.update(self.options._contract_passthrough_options)
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


_MESSAGE_FIELDS = {"role", "content", "name", "tool_call_id", "tool_calls", "reasoning_content"}
_CONTENT_TEXT_FIELDS = {"type", "text", "cache_control"}
_CONTENT_IMAGE_FIELDS = {"type", "url", "detail", "image_url", "cache_control"}
_TOOL_CALL_FIELDS = {"id", "name", "arguments", "index", "extra_content"}
_IMAGE_DETAILS = {"auto", "low", "high"}


def _message_from_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("canonical chat messages must be objects")
    unknown = _unknown_contract_keys(value, _MESSAGE_FIELDS)
    if unknown:
        raise ValueError(f"unsupported canonical message field(s): {', '.join(unknown)}")
    role = value.get("role")
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError("canonical chat messages require a valid role")
    if "content" not in value:
        raise ValueError("canonical chat messages require content")

    result = dict(value)
    result["content"] = _content_from_contract(value["content"])
    if "tool_calls" in value:
        tool_calls = value["tool_calls"]
        if not isinstance(tool_calls, list):
            raise TypeError("canonical message tool_calls must be a list")
        result["tool_calls"] = [_tool_call_from_contract(tool_call) for tool_call in tool_calls]
    return result


def _message_to_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("runtime chat messages must be objects to encode canonical JSON")
    role = value.get("role")
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError("runtime chat messages require a valid role")
    if "content" not in value:
        raise ValueError("runtime chat messages require content")

    result = dict(value)
    result["content"] = _content_to_contract(value["content"])
    if "tool_calls" in value:
        tool_calls = value["tool_calls"]
        if not isinstance(tool_calls, list):
            raise TypeError("runtime message tool_calls must be a list")
        result["tool_calls"] = [_tool_call_to_contract(tool_call) for tool_call in tool_calls]
    return result


def _content_from_contract(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise TypeError("canonical message content must be a string or list")
    return [_content_part_from_contract(part) for part in value]


def _content_to_contract(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise TypeError("runtime message content must be a string or list")
    return [_content_part_to_contract(part) for part in value]


def _content_part_from_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("canonical message content parts must be objects")
    part_type = value.get("type")
    if part_type == "text":
        unknown = _unknown_contract_keys(value, _CONTENT_TEXT_FIELDS)
        if unknown:
            raise ValueError(f"unsupported canonical text content field(s): {', '.join(unknown)}")
        if not isinstance(value.get("text"), str):
            raise ValueError("canonical text content requires text")
        return dict(value)
    if part_type != "image_url":
        raise ValueError(f"unsupported canonical content part type: {part_type!r}")

    unknown = _unknown_contract_keys(value, _CONTENT_IMAGE_FIELDS)
    if unknown:
        raise ValueError(f"unsupported canonical image content field(s): {', '.join(unknown)}")
    has_flat_url = "url" in value
    has_nested_url = "image_url" in value
    if has_flat_url == has_nested_url:
        raise ValueError("canonical image content requires exactly one of url or image_url")

    if has_flat_url:
        url = value.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError("canonical image content requires a non-empty url")
        image_url: dict[str, Any] = {"url": url}
        if "detail" in value:
            image_url["detail"] = _image_detail(value["detail"])
    else:
        nested = value.get("image_url")
        if not isinstance(nested, Mapping):
            raise TypeError("canonical image_url content must be an object")
        if _unknown_contract_keys(nested, {"url", "detail"}):
            unknown_nested = _unknown_contract_keys(nested, {"url", "detail"})
            raise ValueError(f"unsupported canonical image_url field(s): {', '.join(unknown_nested)}")
        url = nested.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError("canonical image_url content requires a non-empty url")
        image_url = dict(nested)
        if "detail" in image_url:
            image_url["detail"] = _image_detail(image_url["detail"])

    result: dict[str, Any] = {"type": "image_url", "image_url": image_url}
    if "cache_control" in value:
        result["cache_control"] = value["cache_control"]
    result.update({key: value[key] for key in value if isinstance(key, str) and key.startswith("x_")})
    return result


def _content_part_to_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("runtime message content parts must be objects")
    part_type = value.get("type")
    if part_type == "text":
        if not isinstance(value.get("text"), str):
            raise ValueError("runtime text content requires text")
        return dict(value)
    if part_type != "image_url":
        raise ValueError(f"unsupported runtime content part type: {part_type!r}")

    nested = value.get("image_url")
    if isinstance(nested, str):
        image_url: dict[str, Any] = {"url": nested}
    elif isinstance(nested, Mapping):
        image_url = dict(nested)
    elif isinstance(value.get("url"), str):
        image_url = {"url": value["url"]}
    else:
        raise ValueError("runtime image content requires image_url or url")

    if "detail" in value and "detail" not in image_url:
        image_url["detail"] = value["detail"]
    if not isinstance(image_url.get("url"), str) or not image_url["url"]:
        raise ValueError("runtime image content requires a non-empty url")
    if "detail" in image_url:
        image_url["detail"] = _image_detail(image_url["detail"])

    result: dict[str, Any] = {"type": "image_url", "image_url": image_url}
    if "cache_control" in value:
        result["cache_control"] = value["cache_control"]
    result.update({key: value[key] for key in value if isinstance(key, str) and key.startswith("x_")})
    return result


def _image_detail(value: Any) -> str:
    if value not in _IMAGE_DETAILS:
        raise ValueError(f"image detail must be one of {sorted(_IMAGE_DETAILS)}")
    return value


def _tool_call_from_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("canonical message tool calls must be objects")
    unknown = _unknown_contract_keys(value, _TOOL_CALL_FIELDS)
    if unknown:
        raise ValueError(f"unsupported canonical tool call field(s): {', '.join(unknown)}")
    call_id = value.get("id")
    name = value.get("name")
    arguments = value.get("arguments")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("canonical tool calls require a non-empty id")
    if not isinstance(name, str) or not name:
        raise ValueError("canonical tool calls require a non-empty name")
    if not isinstance(arguments, str):
        raise TypeError("canonical tool call arguments must be a string")

    function = {"name": name, "arguments": arguments}
    result: dict[str, Any] = {"id": call_id, "type": "function", "function": function}
    if "index" in value:
        if isinstance(value["index"], bool) or not isinstance(value["index"], int) or value["index"] < 0:
            raise ValueError("canonical tool call index must be a non-negative integer")
        result["index"] = value["index"]
    for field in ("extra_content",):
        if field in value:
            result[field] = value[field]
    result.update({key: value[key] for key in value if isinstance(key, str) and key.startswith("x_")})
    return result


def _tool_call_to_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("runtime message tool calls must be objects")
    function = value.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = value.get("name")
        arguments = value.get("arguments")
    call_id = value.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("runtime tool calls require a non-empty id")
    if not isinstance(name, str) or not name:
        raise ValueError("runtime tool calls require a non-empty name")
    if not isinstance(arguments, str):
        raise TypeError("runtime tool call arguments must be a string")

    result: dict[str, Any] = {"id": call_id, "name": name, "arguments": arguments}
    if "index" in value:
        if isinstance(value["index"], bool) or not isinstance(value["index"], int) or value["index"] < 0:
            raise ValueError("runtime tool call index must be a non-negative integer")
        result["index"] = value["index"]
    for field in ("extra_content",):
        if field in value:
            result[field] = value[field]
    result.update({key: value[key] for key in value if isinstance(key, str) and key.startswith("x_")})
    return result


_CONTRACT_REQUEST_FIELDS = {"model", "messages", "options", "tools", "tool_choice", "extra_body"}


def _unknown_contract_keys(value: Mapping[str, Any], allowed: set[str]) -> list[str]:
    return sorted(str(key) for key in value if not isinstance(key, str) or (key not in allowed and not key.startswith("x_")))


def _tool_from_contract(value: Any) -> Any:
    if not isinstance(value, Mapping):
        raise TypeError("canonical chat tools must be objects")
    unknown = _unknown_contract_keys(value, {"name", "description", "parameters", "cache_control"})
    if unknown:
        raise ValueError(f"unsupported canonical tool field(s): {', '.join(unknown)}")
    name = value.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("canonical tools require a non-empty name")
    if "parameters" not in value:
        raise ValueError("canonical tools require parameters")

    function: dict[str, Any] = {"name": name, "parameters": value["parameters"]}
    if "description" in value:
        function["description"] = value["description"]
    result: dict[str, Any] = {"type": "function", "function": function}
    if "cache_control" in value:
        result["cache_control"] = value["cache_control"]
    result.update({key: value[key] for key in value if isinstance(key, str) and key.startswith("x_")})
    return result


def _tool_to_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("runtime chat tools must be objects to encode canonical JSON")

    if value.get("type") == "function" and isinstance(value.get("function"), Mapping):
        function = value["function"]
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("runtime function tools require a non-empty name")
        if "parameters" not in function:
            raise ValueError("runtime function tools require parameters for canonical JSON")
        result: dict[str, Any] = {"name": name, "parameters": function["parameters"]}
        if "description" in function:
            result["description"] = function["description"]
        if "cache_control" in value:
            result["cache_control"] = value["cache_control"]
        result.update({key: value[key] for key in value if isinstance(key, str) and key.startswith("x_")})
        return result

    name = value.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("canonical tools require a non-empty name")
    if "parameters" not in value:
        raise ValueError("canonical tools require parameters")
    allowed = {"name", "description", "parameters", "cache_control"}
    unknown = _unknown_contract_keys(value, allowed)
    if unknown:
        raise ValueError(f"unsupported runtime tool field(s): {', '.join(unknown)}")
    return dict(value)


def _thinking_to_contract(value: Any) -> Any:
    if isinstance(value, ThinkingPreference):
        result: dict[str, Any] = {"type": value.mode.value}
        if value.budget_tokens is not None:
            result["budget_tokens"] = value.budget_tokens
        if value.value is not None:
            result["value"] = dict(value.value)
        return result
    if isinstance(value, Mapping):
        result = dict(value)
        if "type" not in result and "mode" in result:
            result["type"] = result.pop("mode")
        return result
    return value


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
