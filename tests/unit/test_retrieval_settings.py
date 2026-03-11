from __future__ import annotations

import pytest

from vv_llm.settings import Settings
from vv_llm.types.enums import EmbeddingBackendType, RerankBackendType


SETTINGS_PAYLOAD = {
    "VERSION": "2",
    "endpoints": [
        {
            "id": "retrieval-endpoint",
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
                    "endpoints": ["retrieval-endpoint"],
                    "protocol": "openai_embeddings",
                }
            }
        },
        "siliconflow": {
            "models": {
                "Qwen/Qwen3-Embedding-4B": {
                    "id": "Qwen/Qwen3-Embedding-4B",
                    "endpoints": ["retrieval-endpoint"],
                    "protocol": "siliconflow",
                }
            }
        }
    },
    "rerank_backends": {
        "cohere": {
            "models": {
                "rerank-v3.5": {
                    "id": "rerank-v3.5",
                    "endpoints": ["retrieval-endpoint"],
                    "protocol": "cohere_rerank_v2",
                    "default_top_n": 5,
                }
            }
        },
        "siliconflow": {
            "models": {
                "BAAI/bge-reranker-v2-m3": {
                    "id": "BAAI/bge-reranker-v2-m3",
                    "endpoints": ["retrieval-endpoint"],
                    "protocol": "siliconflow",
                }
            }
        }
    },
}


def test_get_embedding_backend() -> None:
    settings = Settings.load_from_dict(SETTINGS_PAYLOAD)

    backend = settings.get_embedding_backend(EmbeddingBackendType.OpenAI)
    assert "text-embedding-3-small" in backend.models
    assert backend.models["text-embedding-3-small"].protocol == "openai_embeddings"


def test_get_rerank_backend() -> None:
    settings = Settings.load_from_dict(SETTINGS_PAYLOAD)

    backend = settings.get_rerank_backend(RerankBackendType.Cohere)
    assert "rerank-v3.5" in backend.models
    assert backend.models["rerank-v3.5"].default_top_n == 5


def test_get_siliconflow_retrieval_backends_with_enums() -> None:
    settings = Settings.load_from_dict(SETTINGS_PAYLOAD)

    embedding_backend = settings.get_embedding_backend(EmbeddingBackendType.Siliconflow)
    rerank_backend = settings.get_rerank_backend(RerankBackendType.Siliconflow)

    assert "Qwen/Qwen3-Embedding-4B" in embedding_backend.models
    assert embedding_backend.models["Qwen/Qwen3-Embedding-4B"].id == "Qwen/Qwen3-Embedding-4B"
    assert embedding_backend.models["Qwen/Qwen3-Embedding-4B"].protocol == "siliconflow"
    assert "BAAI/bge-reranker-v2-m3" in rerank_backend.models
    assert rerank_backend.models["BAAI/bge-reranker-v2-m3"].id == "BAAI/bge-reranker-v2-m3"
    assert rerank_backend.models["BAAI/bge-reranker-v2-m3"].protocol == "siliconflow"


def test_get_missing_retrieval_backend_raises() -> None:
    settings = Settings.load_from_dict(SETTINGS_PAYLOAD)

    with pytest.raises(ValueError, match="Embedding backend jina not found"):
        settings.get_embedding_backend("jina")

    with pytest.raises(ValueError, match="Rerank backend voyage not found"):
        settings.get_rerank_backend("voyage")
