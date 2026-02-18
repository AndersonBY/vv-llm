from __future__ import annotations

from copy import deepcopy

import pytest

from v_llm.settings import Settings, normalize_settings
from v_llm.types.enums import BackendType


def _v2_settings_dict() -> dict:
    return {
        "VERSION": "2",
        "endpoints": [
            {
                "id": "openai-test",
                "api_base": "https://api.openai.com/v1",
                "api_key": "sk-test-key",
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
