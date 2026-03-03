from __future__ import annotations

import json

import httpx
import pytest

from vv_llm.rerank_clients import create_async_rerank_client
from vv_llm.types.enums import RerankBackendType


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
            }
        },
    }


@pytest.mark.asyncio
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
