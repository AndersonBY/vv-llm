from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails

from vv_llm.chat_clients.moonshot_client import AsyncMoonshotChatClient, MoonshotChatClient
from vv_llm.chat_clients.openai_compatible_client import _normalize_openai_compatible_usage
from vv_llm.settings import Settings
from vv_llm.types.enums import BackendType


TEST_MODEL = "kimi-k2.6"
TEST_ENDPOINT = "moonshot-test"


def _completion_usage(**cache_fields: Any) -> CompletionUsage:
    return CompletionUsage(
        completion_tokens=4,
        prompt_tokens=10,
        total_tokens=14,
        **cache_fields,
    )


def _cached_tokens(usage: CompletionUsage, backend: BackendType) -> int | None:
    normalized = _normalize_openai_compatible_usage(usage, backend)
    assert normalized is not None
    if normalized.prompt_tokens_details is None:
        return None
    return normalized.prompt_tokens_details.cached_tokens


def _moonshot_settings() -> Settings:
    return Settings.load_from_dict(
        {
            "VERSION": "2",
            "rate_limit": {"enabled": False},
            "endpoints": [
                {
                    "id": TEST_ENDPOINT,
                    "api_base": "https://api.moonshot.cn/v1",
                    "api_key": "test-key",
                }
            ],
            "backends": {
                "moonshot": {
                    "models": {
                        TEST_MODEL: {
                            "id": TEST_MODEL,
                            "endpoints": [TEST_ENDPOINT],
                            "context_length": 8192,
                            "max_output_tokens": 1024,
                        }
                    }
                }
            },
        }
    )


def _chat_completion(usage: CompletionUsage) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {"content": "ok", "role": "assistant"},
            }
        ],
        created=0,
        model=TEST_MODEL,
        object="chat.completion",
        usage=usage,
    )


def _usage_chunk(usage: CompletionUsage) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chatcmpl-test",
        choices=[],
        created=0,
        model=TEST_MODEL,
        object="chat.completion.chunk",
        usage=usage,
    )


def _bind_raw_client(client: Any, create: Callable[..., Any]) -> None:
    client.endpoint = client.settings.get_endpoint(TEST_ENDPOINT)
    client.model_id = TEST_MODEL
    client.raw_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class _AsyncChunks(AsyncIterator[ChatCompletionChunk]):
    def __init__(self, chunks: list[ChatCompletionChunk]):
        self._chunks = iter(chunks)

    def __aiter__(self) -> _AsyncChunks:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def test_generic_missing_cache_usage_stays_unknown() -> None:
    assert _cached_tokens(_completion_usage(), BackendType.OpenAI) is None


@pytest.mark.parametrize(
    "usage",
    [
        _completion_usage(cached_tokens=0),
        _completion_usage(prompt_tokens_details=PromptTokensDetails(cached_tokens=0)),
    ],
    ids=["top-level", "nested"],
)
def test_generic_explicit_cache_zero_is_preserved(usage: CompletionUsage) -> None:
    assert _cached_tokens(usage, BackendType.OpenAI) == 0


def test_moonshot_omitted_cache_usage_is_normalized_to_zero() -> None:
    assert _cached_tokens(_completion_usage(), BackendType.Moonshot) == 0


@pytest.mark.parametrize(
    "usage",
    [
        _completion_usage(cached_tokens=None),
        _completion_usage(cached_tokens="invalid"),
        _completion_usage(cached_tokens=-1),
        _completion_usage(prompt_tokens_details=None),
        _completion_usage(prompt_tokens_details=PromptTokensDetails()),
        _completion_usage(prompt_tokens_details=PromptTokensDetails(cached_tokens=None)),
        _completion_usage(prompt_tokens_details=PromptTokensDetails(cached_tokens=-1)),
    ],
    ids=["top-level-null", "top-level-string", "top-level-negative", "details-null", "nested-omitted", "nested-null", "nested-negative"],
)
def test_moonshot_invalid_or_null_cache_usage_stays_unknown(usage: CompletionUsage) -> None:
    assert _cached_tokens(usage, BackendType.Moonshot) is None


def test_positive_top_level_cache_usage_is_projected_without_losing_details() -> None:
    raw_usage = _completion_usage(
        cached_tokens=7,
        prompt_tokens_details=PromptTokensDetails(audio_tokens=2, cached_tokens=3),
    )

    normalized = _normalize_openai_compatible_usage(raw_usage, BackendType.Moonshot)

    assert normalized is not None
    assert normalized.prompt_tokens == 10
    assert normalized.prompt_tokens_details is not None
    assert normalized.prompt_tokens_details.cached_tokens == 7
    assert normalized.prompt_tokens_details.audio_tokens == 2


def test_sync_and_async_non_streaming_clients_share_moonshot_normalization() -> None:
    settings = _moonshot_settings()
    response = _chat_completion(_completion_usage())

    sync_client = MoonshotChatClient(model=TEST_MODEL, stream=False, settings=settings)
    _bind_raw_client(sync_client, lambda **kwargs: response)
    sync_result = sync_client.create_completion(
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        max_tokens=16,
    )

    async def run() -> Any:
        async def create(**kwargs: Any) -> ChatCompletion:
            return response

        async_client = AsyncMoonshotChatClient(model=TEST_MODEL, stream=False, settings=settings)
        _bind_raw_client(async_client, create)
        return await async_client.create_completion(
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
            max_tokens=16,
        )

    async_result = asyncio.run(run())

    assert sync_result.usage is not None
    assert async_result.usage is not None
    assert sync_result.usage.prompt_tokens_details is not None
    assert async_result.usage.prompt_tokens_details is not None
    assert sync_result.usage.prompt_tokens_details.cached_tokens == 0
    assert async_result.usage.prompt_tokens_details.cached_tokens == 0


def test_sync_and_async_streaming_clients_preserve_all_zero_usage() -> None:
    settings = _moonshot_settings()
    zero_usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
    chunk = _usage_chunk(zero_usage)

    sync_client = MoonshotChatClient(model=TEST_MODEL, stream=True, settings=settings)
    _bind_raw_client(sync_client, lambda **kwargs: iter([chunk]))
    sync_chunks = list(
        sync_client.create_completion(
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
            max_tokens=16,
        )
    )

    async def run() -> list[Any]:
        async def create(**kwargs: Any) -> _AsyncChunks:
            return _AsyncChunks([chunk])

        async_client = AsyncMoonshotChatClient(model=TEST_MODEL, stream=True, settings=settings)
        _bind_raw_client(async_client, create)
        stream = await async_client.create_completion(
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
            max_tokens=16,
        )
        return [item async for item in stream]

    async_chunks = asyncio.run(run())

    for chunks in (sync_chunks, async_chunks):
        assert len(chunks) == 1
        assert chunks[0].usage is not None
        assert chunks[0].usage.total_tokens == 0
        assert chunks[0].usage.prompt_tokens_details is not None
        assert chunks[0].usage.prompt_tokens_details.cached_tokens == 0
