from __future__ import annotations

from functools import cached_property

import pytest

from vv_llm.chat_clients.base_client import BaseChatClient
from vv_llm.chat_clients.utils import _get_first_enabled_endpoint
from vv_llm.retrieval_clients.common import BaseRetrievalClient
from vv_llm.settings import Settings
from vv_llm.types.enums import BackendType, EmbeddingBackendType


class DummyChatClient(BaseChatClient):
    DEFAULT_MODEL = "gpt-test"
    BACKEND_NAME = BackendType.OpenAI

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.endpoint = getattr(self, "endpoint", None)
        self.model_id = self.backend_settings.models[self.model].id

    @cached_property
    def raw_client(self):
        return None

    def create_completion(self, *args, **kwargs):
        raise NotImplementedError


CHAT_SETTINGS_PAYLOAD = {
    "VERSION": "2",
    "endpoints": [
        {"id": "openai-disabled", "api_base": "https://api.openai.com/v1", "api_key": "sk-1", "enabled": True},
        {"id": "openai-enabled", "api_base": "https://api.openai.com/v1", "api_key": "sk-2", "enabled": True},
    ],
    "backends": {
        "openai": {
            "models": {
                "gpt-test": {
                    "id": "gpt-test",
                    "endpoints": [
                        {"endpoint_id": "openai-disabled", "model_id": "gpt-test-a", "enabled": False},
                        {"endpoint_id": "openai-enabled", "model_id": "gpt-test-b", "enabled": True},
                    ],
                }
            }
        }
    },
}


RETRIEVAL_SETTINGS_PAYLOAD = {
    "VERSION": "2",
    "endpoints": [
        {"id": "retrieval-disabled", "api_base": "https://example.com/v1", "api_key": "sk-1", "enabled": True},
        {"id": "retrieval-enabled", "api_base": "https://example.com/v1", "api_key": "sk-2", "enabled": True},
    ],
    "backends": {},
    "embedding_backends": {
        "openai": {
            "models": {
                "text-embedding-test": {
                    "id": "text-embedding-test",
                    "protocol": "openai_embeddings",
                    "endpoints": [
                        {"endpoint_id": "retrieval-disabled", "model_id": "embed-a", "enabled": False},
                        {"endpoint_id": "retrieval-enabled", "model_id": "embed-b", "enabled": True},
                    ],
                }
            }
        }
    },
}


def test_get_first_enabled_endpoint_skips_disabled_binding() -> None:
    settings = Settings.load_from_dict(CHAT_SETTINGS_PAYLOAD)
    backend_setting = settings.get_backend(BackendType.OpenAI).models["gpt-test"]

    endpoint = _get_first_enabled_endpoint(backend_setting, settings)

    assert endpoint is not None
    assert endpoint.id == "openai-enabled"


def test_chat_client_rejects_explicit_disabled_binding() -> None:
    settings = Settings.load_from_dict(CHAT_SETTINGS_PAYLOAD)
    client = DummyChatClient(model="gpt-test", endpoint_id="openai-disabled", settings=settings)

    with pytest.raises(ValueError, match="disabled for model gpt-test"):
        client._set_endpoint()


def test_chat_client_random_endpoint_skips_disabled_binding() -> None:
    settings = Settings.load_from_dict(CHAT_SETTINGS_PAYLOAD)
    client = DummyChatClient(model="gpt-test", settings=settings)

    endpoint, model_id = client._set_endpoint()

    assert endpoint.id == "openai-enabled"
    assert model_id == "gpt-test-b"


def test_retrieval_client_rejects_explicit_disabled_binding() -> None:
    settings = Settings.load_from_dict(RETRIEVAL_SETTINGS_PAYLOAD)
    backend = settings.get_embedding_backend(EmbeddingBackendType.OpenAI)
    client = BaseRetrievalClient(
        model="text-embedding-test",
        backend_name="openai",
        backend_settings=backend,
        endpoint_id="retrieval-disabled",
        settings=settings,
    )

    with pytest.raises(ValueError, match="disabled for model text-embedding-test"):
        client._set_endpoint()


def test_retrieval_client_random_endpoint_skips_disabled_binding() -> None:
    settings = Settings.load_from_dict(RETRIEVAL_SETTINGS_PAYLOAD)
    backend = settings.get_embedding_backend(EmbeddingBackendType.OpenAI)
    client = BaseRetrievalClient(
        model="text-embedding-test",
        backend_name="openai",
        backend_settings=backend,
        settings=settings,
    )

    endpoint, model_id = client._set_endpoint()

    assert endpoint.id == "retrieval-enabled"
    assert model_id == "embed-b"
