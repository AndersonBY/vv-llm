from __future__ import annotations

import pytest

from vv_llm.settings import Settings
from vv_llm.types.enums import BackendType


def test_settings_populates_default_backend_models() -> None:
    settings = Settings()

    backend = settings.get_backend(BackendType.OpenAI)

    assert settings.VERSION == "2"
    assert backend is settings.backends.openai
    assert backend.models
    assert "openai" not in Settings.model_fields


def test_settings_loads_backend_mapping_without_metadata() -> None:
    settings = Settings.load_from_dict(
        {
            "endpoints": [{"id": "openai-test", "api_key": "sk-test-key"}],
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
    )

    assert settings.get_backend(BackendType.OpenAI).models["gpt-test"].id == "gpt-test"


def test_settings_rejects_top_level_provider_config() -> None:
    with pytest.raises(ValueError, match=r"backends\.openai"):
        Settings.load_from_dict({"openai": {"models": {}}})
