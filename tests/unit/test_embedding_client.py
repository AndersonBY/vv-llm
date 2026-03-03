from __future__ import annotations

from copy import deepcopy
import json

import httpx
import pytest

from vv_llm.embedding_clients import create_embedding_client
from vv_llm.types.enums import EmbeddingBackendType


def _base_settings() -> dict:
    return {
        "VERSION": "2",
        "endpoints": [
            {
                "id": "embed-endpoint",
                "api_base": "https://example.com/v1",
                "api_key": "test-key",
            }
        ],
        "backends": {},
        "embedding_backends": {
            "openai": {
                "models": {
                    "text-embedding-3-small": {
                        "id": "text-embedding-3-small",
                        "endpoints": ["embed-endpoint"],
                        "protocol": "openai_embeddings",
                    }
                }
            },
            "custom": {
                "models": {
                    "custom-embed": {
                        "id": "custom-embed-id",
                        "endpoints": ["embed-endpoint"],
                        "protocol": "custom_json_http",
                        "request_mapping": {
                            "method": "POST",
                            "path": "/custom-embed",
                            "body_template": {
                                "texts": "${input}",
                                "model": "${model_id}",
                            },
                        },
                        "response_mapping": {
                            "model_path": "$.meta.model",
                            "data_path": "$.vectors[*]",
                            "field_map": {
                                "index": "$.i",
                                "embedding": "$.vec",
                            },
                            "usage_map": {
                                "prompt_tokens": "$.meta.usage.prompt_tokens",
                                "total_tokens": "$.meta.usage.total_tokens",
                            },
                        },
                    }
                }
            },
        },
    }


def test_openai_embedding_response_normalization() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "text-embedding-3-small"
        assert payload["input"] == ["a", "b"]
        return httpx.Response(
            status_code=200,
            json={
                "model": "text-embedding-3-small",
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2]},
                    {"index": 1, "embedding": [0.3, 0.4]},
                ],
                "usage": {"prompt_tokens": 8, "total_tokens": 8},
            },
        )

    settings = _base_settings()
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)

    client = create_embedding_client(
        backend=EmbeddingBackendType.OpenAI,
        model="text-embedding-3-small",
        settings=settings,
        http_client=http_client,
    )
    response = client.create_embeddings(input=["a", "b"])

    assert response.model == "text-embedding-3-small"
    assert [d.index for d in response.data] == [0, 1]
    assert response.data[0].embedding == [0.1, 0.2]
    assert response.usage is not None
    assert response.usage.total_tokens == 8


def test_custom_embedding_request_response_mapping() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/custom-embed"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["texts"] == ["hello", "world"]
        assert payload["model"] == "custom-embed-id"

        return httpx.Response(
            status_code=200,
            json={
                "meta": {
                    "model": "mapped-custom-model",
                    "usage": {
                        "prompt_tokens": 5,
                        "total_tokens": 5,
                    },
                },
                "vectors": [
                    {"i": 0, "vec": [1, 2, 3]},
                    {"i": 1, "vec": [4, 5, 6]},
                ],
            },
        )

    settings = _base_settings()
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)

    client = create_embedding_client(
        backend=EmbeddingBackendType.Custom,
        model="custom-embed",
        settings=settings,
        http_client=http_client,
    )
    response = client.create_embeddings(input=["hello", "world"])

    assert response.model == "mapped-custom-model"
    assert len(response.data) == 2
    assert response.data[1].embedding == [4.0, 5.0, 6.0]
    assert response.usage is not None
    assert response.usage.prompt_tokens == 5


def test_custom_embedding_mapping_missing_path_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "meta": {"model": "mapped-custom-model"},
                "vectors": [{"i": 0, "vec": [1, 2, 3]}],
            },
        )

    settings = deepcopy(_base_settings())
    settings["embedding_backends"]["custom"]["models"]["custom-embed"]["response_mapping"]["data_path"] = "$.missing[*]"

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = create_embedding_client(
        backend=EmbeddingBackendType.Custom,
        model="custom-embed",
        settings=settings,
        http_client=http_client,
    )

    with pytest.raises(ValueError, match="response_mapping.data_path"):
        client.create_embeddings(input=["hello"])
