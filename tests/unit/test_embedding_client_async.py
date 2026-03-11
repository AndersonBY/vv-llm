from __future__ import annotations

import json

import httpx
import pytest

from vv_llm.embedding_clients import create_async_embedding_client
from vv_llm.types.enums import EmbeddingBackendType


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


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
            "siliconflow": {
                "models": {
                    "Qwen/Qwen3-Embedding-4B": {
                        "id": "Qwen/Qwen3-Embedding-4B",
                        "endpoints": ["embed-endpoint"],
                        "protocol": "siliconflow",
                    }
                }
            }
        },
    }


@pytest.mark.anyio
async def test_async_openai_embedding_response_normalization() -> None:
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
    http_client = httpx.AsyncClient(transport=transport)

    try:
        client = create_async_embedding_client(
            backend=EmbeddingBackendType.OpenAI,
            model="text-embedding-3-small",
            settings=settings,
            http_client=http_client,
        )
        response = await client.create_embeddings(input=["a", "b"])

        assert response.model == "text-embedding-3-small"
        assert [d.index for d in response.data] == [0, 1]
        assert response.data[0].embedding == [0.1, 0.2]
        assert response.usage is not None
        assert response.usage.total_tokens == 8
    finally:
        await http_client.aclose()


@pytest.mark.anyio
async def test_async_siliconflow_embedding_protocol() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "Qwen/Qwen3-Embedding-4B"
        assert payload["input"] == ["hello", "world"]
        return httpx.Response(
            status_code=200,
            json={
                "model": "Qwen/Qwen3-Embedding-4B",
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2]},
                    {"index": 1, "embedding": [0.3, 0.4]},
                ],
                "usage": {"prompt_tokens": 6, "total_tokens": 6},
            },
        )

    settings = _base_settings()
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)

    try:
        client = create_async_embedding_client(
            backend=EmbeddingBackendType.Siliconflow,
            model="Qwen/Qwen3-Embedding-4B",
            settings=settings,
            http_client=http_client,
        )
        response = await client.create_embeddings(input=["hello", "world"])

        assert response.model == "Qwen/Qwen3-Embedding-4B"
        assert len(response.data) == 2
        assert response.data[0].embedding == [0.1, 0.2]
        assert response.usage is not None
        assert response.usage.total_tokens == 6
    finally:
        await http_client.aclose()
