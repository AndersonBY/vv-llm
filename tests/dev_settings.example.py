"""Template for local credentials.

Usage:
1) Copy this file to `tests/dev_settings.py`
2) Fill in your real keys/secrets
3) Never commit `tests/dev_settings.py`
"""

from __future__ import annotations

from vv_llm.types import SettingsDict

sample_settings: SettingsDict = {
    "VERSION": "2",
    "token_server": {"host": "127.0.0.1", "port": 8338, "url": "http://127.0.0.1:8338"},
    "rate_limit": {
        "enabled": False,
        "backend": "memory",
        "default_rpm": 60,
        "default_tpm": 1000000,
        "redis": {"host": "127.0.0.1", "port": 6379, "db": 0},
        "diskcache": {"cache_dir": ".rate_limit_cache"},
    },
    "endpoints": [
        {
            "id": "openai-default",
            "endpoint_type": "openai",
            "api_base": "https://api.openai.com/v1",
            "api_key": "YOUR_OPENAI_API_KEY",
        },
        {
            "id": "openai-embed-default",
            "endpoint_type": "openai",
            "api_base": "https://api.openai.com/v1",
            "api_key": "YOUR_OPENAI_API_KEY",
        },
        {
            "id": "anthropic-default",
            "endpoint_type": "anthropic",
            "api_base": "https://api.anthropic.com",
            "api_key": "YOUR_ANTHROPIC_API_KEY",
        },
        {
            "id": "gemini-default",
            "endpoint_type": "openai",
            "api_base": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key": "YOUR_GEMINI_API_KEY",
        },
        {
            "id": "cohere-default",
            "endpoint_type": "openai",
            "api_base": "https://api.cohere.com",
            "api_key": "YOUR_COHERE_API_KEY",
        },
        {
            "id": "jina-default",
            "endpoint_type": "openai",
            "api_base": "https://api.jina.ai",
            "api_key": "YOUR_JINA_API_KEY",
        },
        {
            "id": "voyage-default",
            "endpoint_type": "openai",
            "api_base": "https://api.voyageai.com/v1",
            "api_key": "YOUR_VOYAGE_API_KEY",
        },
        {
            "id": "custom-retrieval-default",
            "endpoint_type": "openai",
            "api_base": "https://YOUR_CUSTOM_RETRIEVAL_API_BASE",
            "api_key": "YOUR_CUSTOM_RETRIEVAL_API_KEY",
        },
    ],
    "backends": {
        "openai": {"models": {"gpt-4o": {"id": "gpt-4o", "endpoints": ["openai-default"]}}},
        "anthropic": {"models": {"claude-sonnet-4-6": {"id": "claude-sonnet-4-6", "endpoints": ["anthropic-default"]}}},
        "gemini": {"models": {"gemini-2.5-pro": {"id": "gemini-2.5-pro", "endpoints": ["gemini-default"]}}},
    },
    "embedding_backends": {
        # OpenAI-compatible embeddings
        "openai": {
            "models": {
                "text-embedding-3-small": {
                    "id": "text-embedding-3-small",
                    "endpoints": ["openai-embed-default"],
                    "protocol": "openai_embeddings",
                }
            }
        },
        # Cohere embed v2
        "cohere": {
            "models": {
                "embed-v4.0": {
                    "id": "embed-v4.0",
                    "endpoints": ["cohere-default"],
                    "protocol": "cohere_embed_v2",
                }
            }
        },
        # Voyage embeddings
        "voyage": {
            "models": {
                "voyage-3.5-lite": {
                    "id": "voyage-3.5-lite",
                    "endpoints": ["voyage-default"],
                    "protocol": "voyage_embeddings_v1",
                }
            }
        },
        # Custom JSON HTTP embeddings
        "custom": {
            "models": {
                "custom-embed": {
                    "id": "custom-embed-id",
                    "endpoints": ["custom-retrieval-default"],
                    "protocol": "custom_json_http",
                    "request_mapping": {
                        "method": "POST",
                        "path": "/embed",
                        "body_template": {
                            "model": "${model_id}",
                            "texts": "${input}",
                        },
                    },
                    "response_mapping": {
                        "model_path": "$.meta.model",
                        "data_path": "$.vectors[*]",
                        "field_map": {
                            "index": "$.index",
                            "embedding": "$.vector",
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
    "rerank_backends": {
        # Cohere rerank v2
        "cohere": {
            "models": {
                "rerank-v3.5": {
                    "id": "rerank-v3.5",
                    "endpoints": ["cohere-default"],
                    "protocol": "cohere_rerank_v2",
                    "default_top_n": 10,
                }
            }
        },
        # Jina rerank v1
        "jina": {
            "models": {
                "jina-reranker-v2-base-multilingual": {
                    "id": "jina-reranker-v2-base-multilingual",
                    "endpoints": ["jina-default"],
                    "protocol": "jina_rerank_v1",
                    "default_top_n": 10,
                }
            }
        },
        # Voyage rerank
        "voyage": {
            "models": {
                "rerank-2.5-lite": {
                    "id": "rerank-2.5-lite",
                    "endpoints": ["voyage-default"],
                    "protocol": "voyage_rerank_v1",
                    "default_top_n": 10,
                }
            }
        },
        # Custom JSON HTTP rerank
        "custom": {
            "models": {
                "custom-rerank": {
                    "id": "custom-rerank-id",
                    "endpoints": ["custom-retrieval-default"],
                    "protocol": "custom_json_http",
                    "default_top_n": 10,
                    "request_mapping": {
                        "method": "POST",
                        "path": "/rerank",
                        "body_template": {
                            "model": "${model_id}",
                            "query": "${query}",
                            "documents": "${documents}",
                            "top_n": "${top_n}",
                        },
                    },
                    "response_mapping": {
                        "model_path": "$.meta.model",
                        "results_path": "$.results[*]",
                        "field_map": {
                            "index": "$.index",
                            "relevance_score": "$.score",
                            "document": "$.document",
                        },
                        "usage_map": {
                            "search_units": "$.meta.usage.search_units",
                            "total_tokens": "$.meta.usage.total_tokens",
                        },
                    },
                }
            }
        },
    },
}
