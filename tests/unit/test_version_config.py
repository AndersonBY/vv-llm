from __future__ import annotations

import pytest

from v_llm.settings import Settings
from v_llm.types.enums import BackendType


@pytest.mark.parametrize(
    ("payload", "model_name", "expect_v1_warning"),
    [
        (
            {
                "endpoints": [
                    {
                        "id": "openai-test",
                        "api_base": "https://api.openai.com/v1",
                        "api_key": "sk-test-key",
                    }
                ],
                "openai": {
                    "models": {
                        "gpt-3.5-turbo": {
                            "id": "gpt-3.5-turbo",
                            "endpoints": ["openai-test"],
                        }
                    }
                },
            },
            "gpt-3.5-turbo",
            True,
        ),
        (
            {
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
                            "gpt-4": {
                                "id": "gpt-4",
                                "endpoints": ["openai-test"],
                            }
                        }
                    }
                },
            },
            "gpt-4",
            False,
        ),
    ],
)
def test_settings_support_v1_and_v2_formats(payload: dict, model_name: str, expect_v1_warning: bool) -> None:
    if expect_v1_warning:
        with pytest.warns(UserWarning, match="deprecated V1"):
            settings = Settings(**payload)
    else:
        settings = Settings(**payload)

    backend = settings.get_backend(BackendType.OpenAI)

    assert settings.VERSION == "2"
    assert model_name in backend.models
    assert backend.models[model_name].id == model_name
