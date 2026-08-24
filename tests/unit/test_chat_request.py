from __future__ import annotations

import pytest

from vv_llm import (
    CapabilityPolicy,
    ChatRequest,
    ChatRequestOptions,
    ModelCapabilities,
    ThinkingCapability,
    ThinkingPreference,
)
from vv_llm.types.llm_parameters import BackendSettings, ModelSetting
from vv_llm.chat_clients.tool_call_parser import refactor_tool_choice


def test_thinking_preference_has_distinct_default_enabled_and_disabled_states() -> None:
    assert ThinkingPreference.default().to_provider_value() is None
    assert ThinkingPreference.enabled().to_provider_value() == {"type": "enabled"}
    assert ThinkingPreference.enabled(4096).to_provider_value() == {
        "type": "enabled",
        "budget_tokens": 4096,
    }
    assert ThinkingPreference.disabled().to_provider_value() == {"type": "disabled"}


def test_model_setting_derives_capabilities_from_legacy_flags() -> None:
    setting = ModelSetting(
        id="legacy-model",
        function_call_available=True,
        response_format_available=True,
        native_multimodal=True,
    )

    assert setting.capabilities is not None
    assert setting.capabilities.tools is True
    assert setting.capabilities.structured_output.value == "json_schema"
    assert {modality.value for modality in setting.capabilities.input_modalities} == {"text", "image"}


def test_explicit_capabilities_keep_legacy_properties_compatible() -> None:
    setting = ModelSetting(
        id="typed-model",
        capabilities=ModelCapabilities(
            tools=True,
            structured_output="json_schema",
            input_modalities={"text", "image"},
            thinking=ThinkingCapability.CONFIGURABLE,
        ),
    )

    assert setting.function_call_available is True
    assert setting.response_format_available is True
    assert setting.native_multimodal is True


def test_model_setting_validates_image_dimension_limit() -> None:
    setting = ModelSetting(id="vision-model", max_image_dimension=8192)

    assert setting.max_image_dimension == 8192

    with pytest.raises(ValueError, match="greater than or equal to 1"):
        ModelSetting(id="invalid-vision-model", max_image_dimension=0)


def test_strict_capability_policy_rejects_unsupported_thinking() -> None:
    request = ChatRequest(
        messages=[{"role": "user", "content": "hello"}],
        options=ChatRequestOptions(thinking=ThinkingPreference.enabled()),
    )

    with pytest.raises(ValueError, match="does not support thinking"):
        request.validate_capabilities(
            ModelCapabilities(thinking=ThinkingCapability.UNSUPPORTED),
            CapabilityPolicy.STRICT,
        )


def test_legacy_model_overrides_update_default_capabilities() -> None:
    backend = BackendSettings()
    backend.update_models(
        {
            "model": {
                "id": "model",
                "function_call_available": True,
                "response_format_available": True,
                "native_multimodal": True,
                "capabilities": {
                    "tools": True,
                    "structured_output": "json_schema",
                    "input_modalities": ["text", "image"],
                    "thinking": "configurable",
                },
            }
        },
        {
            "model": {
                "function_call_available": False,
                "response_format_available": False,
                "native_multimodal": False,
            }
        },
    )

    setting = backend.models["model"]
    assert setting.capabilities is not None
    assert setting.capabilities.tools is False
    assert setting.capabilities.structured_output.value == "none"
    assert {modality.value for modality in setting.capabilities.input_modalities} == {"text"}
    assert setting.capabilities.thinking is ThinkingCapability.CONFIGURABLE


def test_contract_codec_maps_nested_stream_and_normalizes_tools() -> None:
    canonical = {
        "model": "contract-model",
        "messages": [{"role": "user", "content": "hello"}],
        "options": {
            "temperature": 0.25,
            "max_tokens": 64,
            "stream": False,
            "thinking": {"type": "disabled"},
            "x_option": {"source": "test"},
        },
        "tools": [
            {
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {"type": "object"},
                "x_tool": "kept",
            }
        ],
        "tool_choice": "auto",
        "extra_body": {"trace_id": "test"},
        "x_request": {"source": "test"},
    }

    request = ChatRequest.from_contract(canonical)

    assert request.stream is False
    assert request.tools == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {"type": "object"},
            },
            "x_tool": "kept",
        }
    ]
    assert request.to_contract() == canonical


def test_contract_codec_requires_model_and_excludes_transport_controls() -> None:
    with pytest.raises(ValueError, match="requires a non-empty model"):
        ChatRequest.from_contract({"messages": []})

    request = ChatRequest(
        model="contract-model",
        messages=[],
        skip_cutoff=True,
        extra_headers={"x-request": "test"},
        header_context={"trace_id": "test"},
        extra_query={"debug": True},
        options=ChatRequestOptions(
            timeout=30,
            provider_options={"deepseek": {"thinking": {"type": "disabled"}}},
        ),
    )

    assert request.to_contract() == {
        "model": "contract-model",
        "messages": [],
        "options": {"stream": False},
    }


def test_contract_codec_normalizes_multimodal_messages_and_tool_calls() -> None:
    canonical = {
        "model": "vision-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look", "cache_control": {"type": "ephemeral"}},
                    {
                        "type": "image_url",
                        "url": "https://example.com/flat.png",
                        "detail": "low",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/nested.png",
                            "detail": "high",
                            "x_image_source": "nested",
                        },
                        "x_part_source": "outer",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "name": "lookup", "arguments": '{"query":"x"}'}],
            },
        ],
    }

    request = ChatRequest.from_contract(canonical)

    assert request.messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look", "cache_control": {"type": "ephemeral"}},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/flat.png", "detail": "low"},
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/nested.png",
                        "detail": "high",
                        "x_image_source": "nested",
                    },
                    "x_part_source": "outer",
                },
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"query":"x"}'},
                }
            ],
        },
    ]
    assert request.to_contract() == {
        **canonical,
        "options": {"stream": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look", "cache_control": {"type": "ephemeral"}},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/flat.png", "detail": "low"},
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/nested.png",
                            "detail": "high",
                            "x_image_source": "nested",
                        },
                        "x_part_source": "outer",
                    },
                ],
            },
            canonical["messages"][1],
        ],
    }


def test_contract_codec_forwards_max_tokens_details_through_extra_body() -> None:
    request = ChatRequest.from_contract(
        {
            "model": "reasoning-model",
            "messages": [{"role": "user", "content": "hello"}],
            "options": {"max_tokens_details": {"reasoning_tokens": 8}},
        }
    )

    assert request.to_completion_kwargs("deepseek")["extra_body"] == {
        "max_tokens_details": {"reasoning_tokens": 8},
    }
    assert request.to_contract()["options"]["max_tokens_details"] == {"reasoning_tokens": 8}


def test_anthropic_tool_choice_rejects_unsupported_objects_without_auto_fallback() -> None:
    assert refactor_tool_choice("auto") == {"type": "auto"}
    assert refactor_tool_choice("required") == {"type": "any"}
    assert refactor_tool_choice({"type": "function", "function": {"name": "lookup"}}) == {"type": "tool", "name": "lookup"}

    with pytest.raises(ValueError, match="requires function.name"):
        refactor_tool_choice({"type": "function", "function": {}})
    with pytest.raises(ValueError, match="must be 'auto'"):
        refactor_tool_choice({"type": "unsupported"})
