from __future__ import annotations

from copy import deepcopy

import pytest

from vv_llm.settings import Settings, normalize_settings
from vv_llm.types.enums import BackendType
from vv_llm.types.llm_parameters import EndpointSetting


def _v2_settings_dict() -> dict:
    return {
        "VERSION": "2",
        "endpoints": [
            {
                "id": "openai-test",
                "api_base": "https://api.openai.com/v1",
                "api_key": "sk-test-key",
                "headers": {
                    "X-Endpoint": "openai-test",
                },
            }
        ],
        "backends": {
            "openai": {
                "models": {
                    "gpt-test": {
                        "id": "gpt-test",
                        "endpoints": ["openai-test"],
                    }
                }
            }
        },
    }


def test_load_from_dict_and_get_backend_model() -> None:
    settings = Settings.load_from_dict(_v2_settings_dict())

    backend = settings.get_backend(BackendType.OpenAI)
    assert "gpt-test" in backend.models
    assert backend.models["gpt-test"].id == "gpt-test"


def test_load_accepts_dict_and_settings_object() -> None:
    settings = Settings()
    payload = _v2_settings_dict()

    settings.load(payload)
    assert settings.get_endpoint("openai-test").api_key == "sk-test-key"
    assert settings.get_endpoint("openai-test").headers == {"X-Endpoint": "openai-test"}

    copied = Settings.load_from_dict(deepcopy(payload))
    settings.load(copied)
    assert settings.get_backend(BackendType.OpenAI).models["gpt-test"].endpoints == ["openai-test"]


def test_normalize_settings_returns_settings_instance() -> None:
    payload = _v2_settings_dict()

    normalized_from_dict = normalize_settings(payload)
    normalized_from_obj = normalize_settings(normalized_from_dict)

    assert isinstance(normalized_from_dict, Settings)
    assert normalized_from_obj is normalized_from_dict


def test_get_endpoint_raises_for_missing_id() -> None:
    settings = Settings.load_from_dict(_v2_settings_dict())

    with pytest.raises(ValueError, match="missing-id"):
        settings.get_endpoint("missing-id")


def test_explicit_endpoint_type_overrides_legacy_azure_flag() -> None:
    settings = Settings.load_from_dict(
        {
            "VERSION": "2",
            "endpoints": [
                {
                    "id": "packy-zhipu",
                    "api_base": "https://www.packyapi.com/v1",
                    "api_key": "sk-test-key",
                    "endpoint_type": "openai",
                    "is_azure": True,
                }
            ],
            "backends": {
                "openai": {
                    "models": {
                        "gpt-test": {
                            "id": "gpt-test",
                            "endpoints": ["packy-zhipu"],
                        }
                    }
                }
            },
        }
    )

    endpoint = settings.get_endpoint("packy-zhipu")

    assert endpoint.endpoint_type == "openai"
    assert endpoint.is_azure is False


def test_legacy_azure_flag_still_maps_when_endpoint_type_missing() -> None:
    settings = Settings.load_from_dict(
        {
            "VERSION": "2",
            "endpoints": [
                {
                    "id": "legacy-azure",
                    "api_base": "https://example.openai.azure.com",
                    "api_key": "sk-test-key",
                    "is_azure": True,
                }
            ],
            "backends": {
                "openai": {
                    "models": {
                        "gpt-test": {
                            "id": "gpt-test",
                            "endpoints": ["legacy-azure"],
                        }
                    }
                }
            },
        }
    )

    endpoint = settings.get_endpoint("legacy-azure")

    assert endpoint.endpoint_type == "openai_azure"
    assert endpoint.is_azure is True


def test_blank_endpoint_proxy_is_normalized_to_none() -> None:
    assert EndpointSetting(id="empty-proxy", proxy="").proxy is None
    assert EndpointSetting(id="blank-proxy", proxy="   ").proxy is None
    assert EndpointSetting(id="configured-proxy", proxy="http://127.0.0.1:7890").proxy == "http://127.0.0.1:7890"
