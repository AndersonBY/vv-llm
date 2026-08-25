# vv-llm

[中文文档](./README_ZH.md)

Universal LLM interface layer for Python. One API, 17 backends, sync & async.

```
pip install vv-llm
```

## Supported Backends

OpenAI | Anthropic | DeepSeek | Gemini | Qwen | Groq | Mistral | Moonshot | MiniMax | Yi | ZhiPuAI | Baichuan | StepFun | xAI | Xiaomi | Ernie | Local

Also supports Azure OpenAI, Vertex AI, and AWS Bedrock deployments.

## Quick Start

### Configure

```python
from vv_llm.settings import settings

settings.load({
    "endpoints": [
        {
            "id": "openai-default",
            "api_base": "https://api.openai.com/v1",
            "api_key": "sk-...",
        }
    ],
    "backends": {
        "openai": {
            "models": {
                "gpt-4o": {
                    "id": "gpt-4o",
                    "endpoints": ["openai-default"],
                }
            }
        }
    }
})
```

### Typed sync (canonical request)

```python
from vv_llm.chat_clients import create_chat_client, BackendType
from vv_llm import ChatRequest, ChatRequestOptions, ThinkingPreference

client = create_chat_client(BackendType.OpenAI, model="gpt-4o")
resp = client.create(
    ChatRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Explain RAG in one sentence"}],
        options=ChatRequestOptions(
            thinking=ThinkingPreference.default(),
            max_tokens=512,
        ),
    )
)
print(resp.content)
```

`ChatRequest` is the normalized runtime request. At a contract boundary,
`ChatRequest.from_contract(...)` decodes canonical JSON (where `model` is
required and `options.stream` is nested), while `to_contract()` omits runtime
transport controls such as headers and query parameters.

Use `ThinkingPreference.default()` to preserve the provider default, `enabled()` or
`enabled(budget_tokens=...)` to opt in, and `disabled()` to opt out explicitly.

### Keyword API

`create_completion(...)` accepts keyword arguments. Pass `thinking` explicitly
when a provider supports Anthropic-style thinking control; omit it to use the
provider default:

```python
resp = client.create_completion(
    messages=[{"role": "user", "content": "Answer directly"}],
    thinking={"type": "disabled"},
)
```

### Middleware, Retry, And Metadata

Wrap a client with `MiddlewareChatClient` for middleware hooks, classified
retry, and execution metadata:

```python
from vv_llm import ChatMiddlewareV1, ChatRequest, MiddlewareChatClient, RetryPolicy

class TraceMiddleware(ChatMiddlewareV1):
    def on_request(self, context, request):
        context.attributes["trace_id"] = "request-42"
        return request

runtime = MiddlewareChatClient(
    client,
    [TraceMiddleware()],
    retry_policy=RetryPolicy(max_attempts=3, total_timeout=20),
)
result = runtime.create_with_metadata(
    ChatRequest(messages=[{"role": "user", "content": "Answer directly"}])
)

print(result.response.content)
print(result.metadata.provider, result.metadata.attempts, result.metadata.latency_ms)
```

`ErrorKind` distinguishes authentication, rate limiting, network, timeout,
invalid request, context length, content policy, missing model, provider
internal, serialization, and configuration failures. The default retry policy
retries only transient kinds and respects `retry-after-ms` plus numeric or
HTTP-date `Retry-After`, exponential backoff, jitter, and an optional total
deadline.

### Explicit Registry And Fallback

Fallback is opt-in and ordered. Every registration declares model capabilities,
so an incompatible route is skipped without sending a request:

```python
from vv_llm import FallbackChatClient, FallbackRoute, ProviderRegistry

registry = ProviderRegistry()
registry.register(
    "primary",
    lambda: primary_client,
    capabilities=primary_client.capabilities,
)
registry.register(
    "secondary",
    lambda: secondary_client,
    capabilities=secondary_client.capabilities,
)
runtime = FallbackChatClient(
    registry,
    [
        FallbackRoute("primary", "primary-model"),
        FallbackRoute("secondary", "secondary-model"),
    ],
)
```

Authentication and invalid-request errors do not fall back by default. Streaming
may switch routes only while establishing the stream or before its first visible
chunk; after output begins, later errors are returned without replay.

### Streaming

```python
from vv_llm import ChatRequest

for chunk in client.create(ChatRequest(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a haiku"}],
    stream=True,
)):
    if chunk.content:
        print(chunk.content, end="")
```

### Async

```python
import asyncio
from vv_llm.chat_clients import create_async_chat_client, BackendType
from vv_llm import ChatRequest

async def main():
    client = create_async_chat_client(BackendType.OpenAI, model="gpt-4o")
    resp = await client.create(ChatRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
    ))
    print(resp.content)

asyncio.run(main())
```

### HTTP transport clients

The `http_client` argument accepts `httpx2.Client` for sync calls and
`httpx2.AsyncClient` for async calls. This lets applications provide a custom
transport (for example, an offline `MockTransport`) while keeping OpenAI 3.x
and Anthropic 1.x clients on the same HTTPX2 runtime:

```python
import httpx2
from vv_llm.chat_clients import BackendType, create_chat_client

transport = httpx2.MockTransport(
    lambda request: httpx2.Response(200, json={"choices": []}, request=request)
)
http_client = httpx2.Client(transport=transport)
client = create_chat_client(BackendType.OpenAI, model="gpt-4o", http_client=http_client)
```

Use an endpoint `proxy` setting when vv-llm should construct the transport
client itself. Legacy `httpx.Client` and `httpx.AsyncClient` instances are not
accepted; use the matching HTTPX2 client type instead.

### Embedding & Rerank

```python
from vv_llm.settings import settings

settings.load({
    "endpoints": [
        {
            "id": "siliconflow",
            "api_base": "https://api.siliconflow.cn/v1",
            "api_key": "sk-...",
        }
    ],
    "backends": {},
    "embedding_backends": {
        "siliconflow": {
            "models": {
                "BAAI/bge-large-zh-v1.5": {
                    "id": "BAAI/bge-large-zh-v1.5",
                    "endpoints": ["siliconflow"],
                    "protocol": "openai_embeddings",
                }
            }
        }
    },
    "rerank_backends": {
        "siliconflow": {
            "models": {
                "BAAI/bge-reranker-v2-m3": {
                    "id": "BAAI/bge-reranker-v2-m3",
                    "endpoints": ["siliconflow"],
                    "protocol": "custom_json_http",
                    "request_mapping": {
                        "method": "POST",
                        "path": "/rerank",
                        "body_template": {
                            "model": "${model_id}",
                            "query": "${query}",
                            "documents": "${documents}",
                        },
                    },
                    "response_mapping": {
                        "results_path": "$.results[*]",
                        "field_map": {
                            "index": "$.index",
                            "relevance_score": "$.relevance_score",
                        },
                    },
                }
            }
        }
    },
})
```

```python
from vv_llm.embedding_clients import create_embedding_client
from vv_llm.rerank_clients import create_rerank_client

embedding_client = create_embedding_client("siliconflow", model="BAAI/bge-large-zh-v1.5")
embedding_resp = embedding_client.create_embeddings(input="hello world")
print(len(embedding_resp.data[0].embedding))

rerank_client = create_rerank_client("siliconflow", model="BAAI/bge-reranker-v2-m3")
rerank_resp = rerank_client.rerank(
    query="Apple",
    documents=["apple", "banana", "fruit", "vegetable"],
)
print(rerank_resp.results[0].index, rerank_resp.results[0].relevance_score)
```

```python
import asyncio
from vv_llm.embedding_clients import create_async_embedding_client
from vv_llm.rerank_clients import create_async_rerank_client

async def main():
    embedding_client = create_async_embedding_client("siliconflow", model="BAAI/bge-large-zh-v1.5")
    rerank_client = create_async_rerank_client("siliconflow", model="BAAI/bge-reranker-v2-m3")

    emb = await embedding_client.create_embeddings(input=["a", "b"])
    rr = await rerank_client.rerank(query="Apple", documents=["apple", "banana"])
    print(len(emb.data), len(rr.results))

asyncio.run(main())
```

## Features

- **Unified interface** — canonical `ChatRequest` execution across all providers, with `create_completion` / `create_stream` retained for compatibility
- **Embedding & rerank** — unified sync/async retrieval clients with normalized outputs
- **Type-safe factory** — `create_chat_client(BackendType.X)` returns the correct client type
- **Multi-endpoint** — configure multiple endpoints per backend with random selection and failover
- **Tool calling** — normalized tool/function calling across providers
- **Multimodal** — text + image inputs where supported
- **Thinking/reasoning** — access chain-of-thought from Claude, DeepSeek Reasoner, etc.
- **Token counting** — per-model tokenizers (tiktoken, deepseek-tokenizer, qwen-tokenizer)
- **Rate limiting** — RPM/TPM controls with memory, Redis, or DiskCache backends
- **Context length control** — automatic message truncation to fit model limits
- **Prompt caching** — Anthropic prompt caching support
- **Retry with backoff** — configurable retry logic for transient failures
- **Versioned middleware** — stable `v1` request, response, and error hooks outside provider adapters
- **Classified errors** — provider-neutral error kinds with retryability and request context
- **Explicit fallback** — registered, ordered, capability-aware routes with no hidden provider switching
- **Scripted testing** — deterministic completion/error/stream scripts for conformance tests

The package includes `vv-llm-contract` 1.0.1. Read contract metadata, the model
catalog, and integrity status through `vv_llm.contract`:

```python
from vv_llm.contract import contract_info, load_catalog, verify_contract

info = contract_info()
assert info.contract_version == "1.0.1"
assert verify_contract().ok
catalog = load_catalog()
```

Maintainers can validate the packaged copy with `pdm run contract-check` and
update it from a verified release directory with
`pdm run contract-sync --source PATH`.

## Python Capability Matrix

| Surface | Python support | Boundary |
|---|---|---|
| Middleware | `MiddlewareChatClient` and `AsyncMiddlewareChatClient`; v1 request/response/error hooks and metadata | Opt-in wrapper around a chat client |
| Fallback | `FallbackChatClient` and `AsyncFallbackChatClient`; ordered, capability-aware routes | Stream fallback is limited to setup/before the first visible chunk |
| Retry | `RetryPolicy` plus sync/async executors; classified transient errors, `Retry-After`, backoff, jitter, deadline | Authentication and invalid-request errors are not retried by default |
| Deterministic testing | Scripted clients, vendored protocol fixtures, and unit tests | No network access; live checks require explicit opt-in |
| Chat providers | Anthropic native adapter; 15 OpenAI-compatible adapters; Local adapter | Sync/async and streaming are normalized; tools, structured output, multimodal input, and thinking remain model/provider dependent |
| Embedding | Sync/async configured clients | `openai_embeddings`, SiliconFlow, Cohere, Voyage, and custom JSON HTTP protocols |
| Rerank | Sync/async configured clients | OpenAI-compatible, Cohere, Jina, Voyage, SiliconFlow, and custom JSON HTTP protocols |

### Chat Provider Matrix

| Adapter | Providers | Transport and common behavior |
|---|---|---|
| Native | Anthropic | Native sync/async chat, streaming, tools, vision, thinking, and prompt-cache handling |
| OpenAI-compatible | OpenAI, DeepSeek, Gemini, Groq, MiniMax, Mistral, Moonshot, Qwen, Yi, ZhiPuAI, Baichuan, StepFun, xAI, Xiaomi, Ernie | Shared sync/async request and stream normalization; actual tools, structured output, multimodal, and reasoning support follows the vendored model catalog and provider endpoint |
| Local | Local | Same configured adapter shape for sync/async and streaming; endpoint behavior is deployment-specific |

## Examples

Runnable examples are in [`examples/`](examples/README.md): `basic_chat.py`,
`streaming.py`, `tools.py`, `multimodal.py`, and `contract_json.py` cover the
main typed request paths. `async_streaming.py`, `typed_thinking.py`,
`middleware_metadata.py`, and `registry_fallback.py` cover focused extensions;
the last one is deterministic and offline.

## Cache Usage Semantics

OpenAI-compatible chat completions report cache reads through `usage.prompt_tokens_details.cached_tokens`. `usage.prompt_tokens` remains the total input token count, so consumers can calculate uncached input as `prompt_tokens - cached_tokens`. This path intentionally does not populate Anthropic's `cache_read_input_tokens` field because Anthropic defines its base `input_tokens` as uncached input.

For generic OpenAI-compatible backends, omitted cache-read fields remain unknown, while an explicit `cached_tokens: 0` is preserved as an observed zero. Moonshot may omit both top-level `cached_tokens` and `prompt_tokens_details` on a cold request; only in that fully omitted case does vv-llm project `prompt_tokens_details.cached_tokens = 0` from the provider contract. Explicit `null` or invalid cache values remain unknown.

## Utilities

```python
from vv_llm.chat_clients import format_messages, get_token_counts, get_message_token_counts
```

| Function | Description |
|---|---|
| `format_messages` | Normalize multimodal/tool messages across formats |
| `get_token_counts` | Count tokens for a text string |
| `get_message_token_counts` | Count tokens for a message list |

## Optional Dependencies

```bash
pip install 'vv-llm[redis]'      # Redis rate limiting
pip install 'vv-llm[diskcache]'  # DiskCache rate limiting
pip install 'vv-llm[server]'     # FastAPI token server
pip install 'vv-llm[vertex]'     # Google Vertex AI
pip install 'vv-llm[bedrock]'    # AWS Bedrock
```

## Project Structure

```
src/vv_llm/
  _contract/      # Versioned schemas, fixtures, catalog, and consumer lock
  chat_clients/    # Per-backend clients + factory
  embedding_clients/  # Embedding clients + factory
  rerank_clients/     # Rerank clients + factory
  retrieval_clients/  # Shared retrieval client internals
  settings/        # Configuration management
  types/           # Type definitions & enums
  utilities/       # Rate limiting, retry, media processing, token counting
  server/          # Optional token counting server

tests/unit/        # Unit tests
tests/live/        # Live integration tests (requires real API keys)
```

## User, Maintainer, And Release Workflows

### Users

Install the package and configure endpoints through the public `Settings` API.
Contract artifacts are read from the installed package; no repository checkout
or local path is required at runtime.

### Maintainers

```bash
pdm install -d          # Install dev dependencies
pdm run contract-check  # Validate only the vendored contract lock
pdm run contract-sync --source PATH  # Refresh from an explicit source tree
# Or: VV_LLM_CONTRACT_SOURCE=PATH pdm run contract-sync
# Compare an explicit source tree with the vendor:
python scripts/sync_contract.py --check --source PATH
pdm run lint            # Ruff linter
pdm run format-check    # Ruff format check
pdm run type-check      # Ty type checker
pdm run test            # Unit tests
```

For an intentional live smoke check, provide private settings through the
existing `tests/dev_settings.py` mechanism and opt in explicitly:

```bash
VV_LLM_RUN_LIVE_TESTS=1 python tests/live/run_live_tests.py test_deepseek_contract_smoke.py
```

The smoke output contains only provider/model, response shape, usage counters,
and exit status; it does not print credentials or response content.

### Release publishers

```bash
pdm build
python scripts/smoke_wheel.py
```

The release CI performs contract-check, unit tests, linting, package build, and
isolated wheel smoke before publication. Live API checks are intentionally not
part of release CI.

## License

MIT
