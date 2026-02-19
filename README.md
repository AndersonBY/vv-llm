# vv-llm

[中文文档](./README_ZH.md)

Universal LLM interface layer for Python. One API, 16 backends, sync & async.

```
pip install vv-llm
```

## Supported Backends

OpenAI | Anthropic | DeepSeek | Gemini | Qwen | Groq | Mistral | Moonshot | MiniMax | Yi | ZhiPuAI | Baichuan | StepFun | xAI | Ernie | Local

Also supports Azure OpenAI, Vertex AI, and AWS Bedrock deployments.

## Quick Start

### Configure

```python
from vv_llm.settings import settings

settings.load({
    "VERSION": "2",
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

### Sync

```python
from vv_llm.chat_clients import create_chat_client, BackendType

client = create_chat_client(BackendType.OpenAI, model="gpt-4o")
resp = client.create_completion([
    {"role": "user", "content": "Explain RAG in one sentence"}
])
print(resp.content)
```

### Streaming

```python
for chunk in client.create_stream([
    {"role": "user", "content": "Write a haiku"}
]):
    if chunk.content:
        print(chunk.content, end="")
```

### Async

```python
import asyncio
from vv_llm.chat_clients import create_async_chat_client, BackendType

async def main():
    client = create_async_chat_client(BackendType.OpenAI, model="gpt-4o")
    resp = await client.create_completion([
        {"role": "user", "content": "hello"}
    ])
    print(resp.content)

asyncio.run(main())
```

## Features

- **Unified interface** — same `create_completion` / `create_stream` API across all providers
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
  chat_clients/    # Per-backend clients + factory
  settings/        # Configuration management
  types/           # Type definitions & enums
  utilities/       # Rate limiting, retry, media processing, token counting
  server/          # Optional token counting server

tests/unit/        # Unit tests
tests/live/        # Live integration tests (requires real API keys)
```

## Development

```bash
pdm install -d          # Install dev dependencies
pdm run lint            # Ruff linter
pdm run format-check    # Ruff format check
pdm run type-check      # Ty type checker
pdm run test            # Unit tests
pdm run test-live       # Live tests (needs real endpoints)
```

## License

MIT
