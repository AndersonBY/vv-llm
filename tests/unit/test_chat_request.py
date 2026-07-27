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
