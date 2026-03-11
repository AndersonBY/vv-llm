from __future__ import annotations

import json

import httpx
import pytest

import vv_llm.retrieval_clients.common as retrieval_common
from vv_llm.rerank_clients import create_async_rerank_client
from vv_llm.types.enums import RerankBackendType


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _base_settings() -> dict:
    return {
        "VERSION": "2",
        "endpoints": [
            {
                "id": "rerank-endpoint",
                "api_base": "https://example.com/v1",
                "api_key": "test-key",
            }
        ],
        "backends": {},
        "rerank_backends": {
            "cohere": {
                "models": {
                    "rerank-v3.5": {
                        "id": "rerank-v3.5",
                        "endpoints": ["rerank-endpoint"],
                        "protocol": "cohere_rerank_v2",
                        "default_top_n": 2,
                    }
                }
            },
            "siliconflow": {
                "models": {
                    "BAAI/bge-reranker-v2-m3": {
                        "id": "BAAI/bge-reranker-v2-m3",
                        "endpoints": ["rerank-endpoint"],
                        "protocol": "siliconflow",
                    }
                }
            }
        },
    }


@pytest.mark.anyio
async def test_async_cohere_rerank_response_normalization() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/v2/rerank"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "rerank-v3.5"
        assert payload["top_n"] == 2

        return httpx.Response(
            status_code=200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.92, "document": {"text": "doc2"}},
                    {"index": 0, "relevance_score": 0.60, "document": {"text": "doc1"}},
                ],
                "meta": {
                    "billed_units": {
                        "search_units": 1,
                    }
                },
            },
        )

    settings = _base_settings()
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)

    try:
        client = create_async_rerank_client(
            backend=RerankBackendType.Cohere,
            model="rerank-v3.5",
            settings=settings,
            http_client=http_client,
        )
        response = await client.rerank(query="q", documents=["doc1", "doc2"])

        assert len(response.results) == 2
        assert response.results[0].index == 1
        assert response.results[0].relevance_score == 0.92
        assert response.results[0].document == "doc2"
        assert response.usage is not None
        assert response.usage.search_units == 1
    finally:
        await http_client.aclose()


@pytest.mark.anyio
async def test_async_siliconflow_rerank_protocol() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/rerank"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload == {
            "model": "BAAI/bge-reranker-v2-m3",
            "query": "apple",
            "documents": ["banana", "apple"],
        }

        return httpx.Response(
            status_code=200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.97, "document": {"text": "apple"}},
                ],
                "meta": {
                    "billed_units": {"search_units": 3},
                    "tokens": {"input_tokens": 18},
                },
            },
        )

    settings = _base_settings()
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)

    try:
        client = create_async_rerank_client(
            backend=RerankBackendType.Siliconflow,
            model="BAAI/bge-reranker-v2-m3",
            settings=settings,
            http_client=http_client,
        )
        response = await client.rerank(query="apple", documents=["banana", "apple"])

        assert response.model == "BAAI/bge-reranker-v2-m3"
        assert len(response.results) == 1
        assert response.results[0].index == 1
        assert response.results[0].document == "apple"
        assert response.usage is not None
        assert response.usage.search_units == 3
        assert response.usage.total_tokens == 18
    finally:
        await http_client.aclose()


@pytest.mark.anyio
async def test_async_rerank_retries_transient_503(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(status_code=503, json={"error": "temporary unavailable"})
        return httpx.Response(
            status_code=200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.88, "document": {"text": "doc-a"}},
                ],
                "meta": {
                    "billed_units": {"search_units": 2},
                    "tokens": {"input_tokens": 9},
                },
            },
        )

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(retrieval_common, "_compute_retry_delay_seconds", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(retrieval_common.asyncio, "sleep", _no_sleep)

    settings = _base_settings()
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)

    try:
        client = create_async_rerank_client(
            backend=RerankBackendType.Siliconflow,
            model="BAAI/bge-reranker-v2-m3",
            settings=settings,
            http_client=http_client,
        )
        response = await client.rerank(query="apple", documents=["doc-a"])

        assert attempts == 2
        assert response.model == "BAAI/bge-reranker-v2-m3"
        assert response.results[0].document == "doc-a"
        assert response.usage is not None
        assert response.usage.total_tokens == 9
    finally:
        await http_client.aclose()
