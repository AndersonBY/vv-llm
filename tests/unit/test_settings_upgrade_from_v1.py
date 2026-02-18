from __future__ import annotations

import pytest

from vv_llm.settings import Settings
from vv_llm.types.enums import BackendType


V1_SETTINGS = {
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
}


V2_SETTINGS = {
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
}


def test_upgrade_v1_to_v2_moves_backend_to_backends() -> None:
    with pytest.warns(UserWarning, match="deprecated V1"):
        settings = Settings(**V1_SETTINGS)

    upgraded = settings.upgrade_to_v2()

    assert upgraded is settings
    assert upgraded.VERSION == "2"
    assert upgraded.backends is not None
    assert upgraded.get_backend(BackendType.OpenAI).models["gpt-3.5-turbo"].id == "gpt-3.5-turbo"


def test_upgrade_v2_is_noop() -> None:
    settings = Settings(**V2_SETTINGS)

    upgraded = settings.upgrade_to_v2()

    assert upgraded is settings
    assert upgraded.VERSION == "2"
    assert upgraded.get_backend(BackendType.OpenAI).models["gpt-4"].id == "gpt-4"
