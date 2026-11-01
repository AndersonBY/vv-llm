from __future__ import annotations

import asyncio
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pytest
from openai.types.chat import ChatCompletionChunk

from vv_llm.chat_clients.moonshot_client import AsyncMoonshotChatClient, MoonshotChatClient
from vv_llm.settings import Settings


MODEL = "kimi-k2.6"
ENDPOINT = "moonshot-test"


def _settings() -> Settings:
    return Settings.load_from_dict(
        {
            "rate_limit": {"enabled": False},
            "endpoints": [{"id": ENDPOINT, "api_base": "https://api.moonshot.cn/v1", "api_key": "test-key"}],
            "backends": {
                "moonshot": {
                    "models": {
                        MODEL: {
                            "id": MODEL,
                            "endpoints": [ENDPOINT],
                            "context_length": 8192,
                            "max_output_tokens": 1024,
                            "function_call_available": True,
                        }
                    }
                }
            },
        }
    )


def _chunk(tool_calls: list[dict[str, Any]]) -> ChatCompletionChunk:
    # Like the SDK response parser, construct permits nonconforming providers
    # to omit index or return null. Validating would reject these before our fix.
    return ChatCompletionChunk.construct(
        id="chatcmpl-test",
        created=0,
        model=MODEL,
        object="chat.completion.chunk",
        choices=[{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": None}],
    )


def _stream(chunks: list[ChatCompletionChunk], asynchronous: bool) -> list[Any]:
    settings = _settings()
    client_class = AsyncMoonshotChatClient if asynchronous else MoonshotChatClient
    client = client_class(model=MODEL, stream=True, settings=settings)
    client.endpoint = settings.get_endpoint(ENDPOINT)
    client.model_id = MODEL

    async def async_chunks():
        for chunk in chunks:
            yield chunk

    async def async_create(**kwargs):
        return async_chunks()

    create = async_create if asynchronous else lambda **kwargs: iter(chunks)
    client.raw_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    kwargs = {"messages": [{"role": "user", "content": "hello"}], "stream": True, "max_tokens": 16}
    if asynchronous:

        async def collect():
            response = await client.create_completion(**kwargs)
            return [message async for message in response]

        return asyncio.run(collect())
    return list(client.create_completion(**kwargs))


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_interleaved_tool_calls_keep_identity_and_arguments(asynchronous: bool) -> None:
    chunks = [
        _chunk([{"index": 0, "id": "call_a", "type": "function", "function": {"name": "f", "arguments": '{"x":'}}]),
        _chunk([{"index": 1, "id": "call_b", "type": "function", "function": {"name": "g", "arguments": '{"y":'}}]),
        _chunk([{"index": 0, "function": {"arguments": "1}"}}]),
        _chunk([{"index": 1, "function": {"arguments": "2}"}}]),
    ]
    calls = [call for message in _stream(chunks, asynchronous) for call in message.tool_calls or []]
    assert [call.index for call in calls] == [0, 1, 0, 1]
    assert [call.id for call in calls] == ["call_a", "call_b", None, None]
    arguments = defaultdict(str)
    for call in calls:
        arguments[call.index] += call.function.arguments or ""
        assert call.type == "function"
    assert dict(arguments) == {0: '{"x":1}', 1: '{"y":2}'}


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_sparse_out_of_order_tool_indices_are_not_chunk_positions(asynchronous: bool) -> None:
    # Explicit zero at position 1 catches accidental truthiness-based fallback.
    chunks = [_chunk([{"index": index, "function": {"arguments": ""}} for index in [7, 0, 3]])]
    calls = [call for message in _stream(chunks, asynchronous) for call in message.tool_calls or []]
    assert [call.index for call in calls] == [7, 0, 3]
    assert all(call.type == "function" for call in calls)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_missing_and_null_tool_indices_keep_legacy_positional_fallback(asynchronous: bool) -> None:
    chunks = [
        _chunk(
            [
                {"index": 7, "function": {"arguments": ""}},
                {"function": {"arguments": ""}},
                {"index": None, "type": None, "function": {"arguments": ""}},
            ]
        )
    ]
    calls = [call for message in _stream(chunks, asynchronous) for call in message.tool_calls or []]
    assert [call.index for call in calls] == [7, 1, 2]
    assert all(call.type == "function" for call in calls)
