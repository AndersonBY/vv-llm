from __future__ import annotations

import asyncio

import pytest

from vv_llm import (
    AsyncScriptedChatClient,
    ChatRequest,
    ErrorKind,
    MiddlewareChatClient,
    RetryPolicy,
    ScriptedChatClient,
    ScriptedStream,
    VvLlmError,
)


def test_scripted_client_drives_retry_contract_and_records_requests() -> None:
    scripted = ScriptedChatClient(
        [
            VvLlmError("temporary", kind=ErrorKind.PROVIDER_INTERNAL),
            "ok",
        ],
        provider="unit",
        model="unit-model",
    )
    client = MiddlewareChatClient(
        scripted,
        retry_policy=RetryPolicy(max_attempts=2, base_delay=0, jitter_ratio=0),
    )

    result = client.create_with_metadata(
        ChatRequest(messages=[{"role": "user", "content": "hello"}])
    )

    assert result.response == "ok"
    assert result.metadata.attempts == 2
    assert len(scripted.requests) == 2
    assert scripted.requests[0].messages[0]["content"] == "hello"


def test_scripted_stream_can_fail_after_visible_output_without_replay() -> None:
    client = ScriptedChatClient(
        [ScriptedStream(["first", VvLlmError("late failure", kind=ErrorKind.NETWORK)])]
    )

    stream = client.create_stream(messages=[{"role": "user", "content": "hello"}])
    assert next(stream) == "first"
    with pytest.raises(VvLlmError, match="late failure"):
        next(stream)
    assert len(client.requests) == 1


def test_async_scripted_client_uses_the_same_steps() -> None:
    async def run() -> tuple[str, int]:
        client = AsyncScriptedChatClient(["ok"])
        response = await client.create(
            ChatRequest(messages=[{"role": "user", "content": "hello"}])
        )
        return response, len(client.requests)

    assert asyncio.run(run()) == ("ok", 1)
