from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from vv_llm import (
    AsyncMiddlewareChatClient,
    ChatMiddlewareV1,
    ChatRequest,
    ErrorKind,
    MiddlewareChatClient,
    RetryPolicy,
    VvLlmError,
)


class RecordingMiddleware(ChatMiddlewareV1):
    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []

    def on_request(self, context: Any, request: ChatRequest) -> ChatRequest:
        self.events.append(("request", context.attempt))
        context.attributes["trace_id"] = "trace-1"
        return request.model_copy(update={"model": "rewritten"})

    def on_response(self, context: Any, response: Any) -> Any:
        self.events.append(("response", context.attempt))
        return {"response": response, "trace_id": context.attributes["trace_id"]}

    def on_error(self, context: Any, error: VvLlmError) -> None:
        self.events.append((error.kind.value, context.attempt))


class FlakyClient:
    backend_name = SimpleNamespace(value="test")
    model = "original"

    def __init__(self) -> None:
        self.attempts = 0

    def create(self, request: ChatRequest, **_: Any) -> str:
        self.attempts += 1
        assert request.model == "rewritten"
        if self.attempts == 1:
            raise VvLlmError("temporary", kind=ErrorKind.PROVIDER_INTERNAL)
        return "ok"


def test_sync_middleware_wraps_normalized_and_legacy_calls() -> None:
    middleware = RecordingMiddleware()
    client = MiddlewareChatClient(
        FlakyClient(),
        [middleware],
        retry_policy=RetryPolicy(max_attempts=2, base_delay=0, jitter_ratio=0),
    )

    response = client.create_completion(
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.2,
    )

    assert response == {"response": "ok", "trace_id": "trace-1"}
    assert middleware.events == [
        ("request", 0),
        ("provider_internal", 1),
        ("response", 2),
    ]


def test_middleware_can_return_response_metadata_without_changing_complete() -> None:
    middleware = RecordingMiddleware()
    client = MiddlewareChatClient(
        FlakyClient(),
        [middleware],
        retry_policy=RetryPolicy(max_attempts=2, base_delay=0, jitter_ratio=0),
    )

    result = client.create_with_metadata(
        ChatRequest(messages=[{"role": "user", "content": "hello"}])
    )

    assert result.response == {"response": "ok", "trace_id": "trace-1"}
    assert result.metadata.provider == "test"
    assert result.metadata.model == "rewritten"
    assert result.metadata.attempts == 2
    assert result.metadata.latency_ms is not None


def test_unknown_middleware_version_is_rejected() -> None:
    middleware = RecordingMiddleware()
    middleware.api_version = "v2"

    try:
        MiddlewareChatClient(FlakyClient(), [middleware])
    except ValueError as error:
        assert "v2" in str(error)
    else:
        raise AssertionError("middleware version should be validated")


def test_async_middleware_uses_the_same_contract() -> None:
    class AsyncClient(FlakyClient):
        async def create(self, request: ChatRequest, **kwargs: Any) -> str:
            return super().create(request, **kwargs)

    async def run() -> tuple[Any, list[tuple[str, int]]]:
        middleware = RecordingMiddleware()
        client = AsyncMiddlewareChatClient(
            AsyncClient(),
            [middleware],
            retry_policy=RetryPolicy(max_attempts=2, base_delay=0, jitter_ratio=0),
        )
        response = await client.create(
            ChatRequest(messages=[{"role": "user", "content": "hello"}])
        )
        return response, middleware.events

    response, events = asyncio.run(run())
    assert response == {"response": "ok", "trace_id": "trace-1"}
    assert events[-1] == ("response", 2)
