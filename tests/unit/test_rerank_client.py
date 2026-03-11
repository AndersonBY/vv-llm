from __future__ import annotations

from copy import deepcopy
import json

import httpx
import pytest

from vv_llm.rerank_clients import create_rerank_client
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
            },
            "siliconflow": {
                "models": {
                    "BAAI/bge-reranker-v2-m3": {
                        "id": "BAAI/bge-reranker-v2-m3",
                        "endpoints": ["rerank-endpoint"],
                        "protocol": "siliconflow",
                    }
                }
            },
            "custom": {
                "models": {
                    "custom-rerank": {
                        "id": "custom-rerank-id",
                        "endpoints": ["rerank-endpoint"],
                        "protocol": "custom_json_http",
                        "request_mapping": {
                            "method": "POST",
                            "path": "/rank",
                            "body_template": {
                                "q": "${query}",
                                "docs": "${documents}",
                                "limit": "${top_n}",
                                "model": "${model_id}",
                            },
                        },
                        "response_mapping": {
                            "model_path": "$.meta.model",
                            "results_path": "$.items[*]",
                            "field_map": {
                                "index": "$.pos",
                                "relevance_score": "$.score",
                                "document": "$.doc",
                            },
                            "usage_map": {
                                "search_units": "$.meta.search_units",
                            },
                        },
                    }
                }
            },
        },
    }


def test_cohere_rerank_response_normalization() -> None:
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
    http_client = httpx.Client(transport=transport)

    client = create_rerank_client(
        backend=RerankBackendType.Cohere,
        model="rerank-v3.5",
        settings=settings,
        http_client=http_client,
    )
    response = client.rerank(query="q", documents=["doc1", "doc2"])

    assert len(response.results) == 2
    assert response.results[0].index == 1
    assert response.results[0].relevance_score == 0.92
    assert response.results[0].document == "doc2"
    assert response.usage is not None
    assert response.usage.search_units == 1


def test_custom_rerank_request_response_mapping() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/rank"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["q"] == "hello"
        assert payload["limit"] == 1
        assert payload["model"] == "custom-rerank-id"

        return httpx.Response(
            status_code=200,
            json={
                "meta": {
                    "model": "mapped-rerank-model",
                    "search_units": 7,
                },
                "items": [
                    {"pos": 0, "score": 0.88, "doc": "doc-a"},
                ],
            },
        )

    settings = _base_settings()
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)

    client = create_rerank_client(
        backend=RerankBackendType.Custom,
        model="custom-rerank",
        settings=settings,
        http_client=http_client,
    )
    response = client.rerank(query="hello", documents=["doc-a", "doc-b"], top_n=1)

    assert response.model == "mapped-rerank-model"
    assert len(response.results) == 1
    assert response.results[0].index == 0
    assert response.results[0].document == "doc-a"
    assert response.usage is not None
    assert response.usage.search_units == 7


def test_siliconflow_rerank_protocol() -> None:
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
                    {"index": 0, "relevance_score": 0.21, "document": {"text": "banana"}},
                ],
                "meta": {
                    "billed_units": {"search_units": 3},
                    "tokens": {"input_tokens": 18},
                },
            },
        )

    settings = _base_settings()
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)

    client = create_rerank_client(
        backend=RerankBackendType.Siliconflow,
        model="BAAI/bge-reranker-v2-m3",
        settings=settings,
        http_client=http_client,
    )
    response = client.rerank(query="apple", documents=["banana", "apple"])

    assert response.model == "BAAI/bge-reranker-v2-m3"
    assert len(response.results) == 2
    assert response.results[0].index == 1
    assert response.results[0].document == "apple"
    assert response.usage is not None
    assert response.usage.search_units == 3
    assert response.usage.total_tokens == 18


def test_custom_rerank_mapping_missing_score_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "meta": {"model": "mapped-rerank-model", "search_units": 7},
                "items": [{"pos": 0, "doc": "doc-a"}],
            },
        )

    settings = deepcopy(_base_settings())
    settings["rerank_backends"]["custom"]["models"]["custom-rerank"]["response_mapping"]["field_map"]["relevance_score"] = "$.missing_score"

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = create_rerank_client(
        backend=RerankBackendType.Custom,
        model="custom-rerank",
        settings=settings,
        http_client=http_client,
    )

    with pytest.raises(ValueError, match="field_map.relevance_score"):
        client.rerank(query="hello", documents=["doc-a"], top_n=1)
