from __future__ import annotations

import asyncio

import pytest

from vv_llm import (
    AsyncFallbackChatClient,
    AsyncScriptedChatClient,
    ChatRequest,
    ErrorKind,
    FallbackChatClient,
    FallbackRoute,
    ModelCapabilities,
    ProviderRegistry,
    ScriptedChatClient,
    ScriptedStream,
    VvLlmError,
)


def test_fallback_is_explicit_and_skips_incompatible_candidates() -> None:
    registry = ProviderRegistry()
    incapable = ScriptedChatClient(["must not run"], provider="incapable")
    capable = ScriptedChatClient(["ok"], provider="capable")
    registry.register(
        "incapable",
        lambda: incapable,
        capabilities=ModelCapabilities(tools=False),
    )
    registry.register(
        "capable",
        lambda: capable,
        capabilities=ModelCapabilities(tools=True),
    )
    client = FallbackChatClient(
        registry,
        [
            FallbackRoute("incapable", "model-a"),
            FallbackRoute("capable", "model-b"),
        ],
    )

    result = client.create_with_metadata(
        ChatRequest(
            messages=[{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
        )
    )

    assert result.response == "ok"
    assert result.metadata.provider == "capable"
    assert result.metadata.fallback_index == 1
    assert incapable.requests == ()
    assert capable.requests[0].model == "model-b"


def test_fallback_does_not_hide_authentication_errors() -> None:
    registry = ProviderRegistry()
    primary = ScriptedChatClient(
        [VvLlmError("unauthorized", kind=ErrorKind.AUTHENTICATION)]
    )
    secondary = ScriptedChatClient(["must not run"])
    capabilities = ModelCapabilities()
    registry.register("primary", lambda: primary, capabilities=capabilities)
    registry.register("secondary", lambda: secondary, capabilities=capabilities)
    client = FallbackChatClient(
        registry,
        [
            FallbackRoute("primary", "model-a"),
            FallbackRoute("secondary", "model-b"),
        ],
    )

    with pytest.raises(VvLlmError) as caught:
        client.create(ChatRequest(messages=[{"role": "user", "content": "hello"}]))

    assert caught.value.kind is ErrorKind.AUTHENTICATION
    assert secondary.requests == ()


def test_stream_fallback_stops_after_first_visible_chunk() -> None:
    registry = ProviderRegistry()
    primary = ScriptedChatClient(
        [
            ScriptedStream(
                [
                    "first",
                    VvLlmError("late failure", kind=ErrorKind.NETWORK),
                ]
            )
        ]
    )
    secondary = ScriptedChatClient([ScriptedStream(["secondary"])])
    capabilities = ModelCapabilities(streaming=True)
    registry.register("primary", lambda: primary, capabilities=capabilities)
    registry.register("secondary", lambda: secondary, capabilities=capabilities)
    client = FallbackChatClient(
        registry,
        [
            FallbackRoute("primary", "model-a"),
            FallbackRoute("secondary", "model-b"),
        ],
    )

    stream = client.create(
        ChatRequest(
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
    )
    assert next(stream) == "first"
    with pytest.raises(VvLlmError, match="late failure"):
        next(stream)
    assert secondary.requests == ()


def test_stream_can_fallback_when_first_chunk_is_an_error() -> None:
    registry = ProviderRegistry()
    primary = ScriptedChatClient(
        [ScriptedStream([VvLlmError("early failure", kind=ErrorKind.NETWORK)])]
    )
    secondary = ScriptedChatClient([ScriptedStream(["secondary"])])
    capabilities = ModelCapabilities(streaming=True)
    registry.register("primary", lambda: primary, capabilities=capabilities)
    registry.register("secondary", lambda: secondary, capabilities=capabilities)
    client = FallbackChatClient(
        registry,
        [
            FallbackRoute("primary", "model-a"),
            FallbackRoute("secondary", "model-b"),
        ],
    )

    stream = client.create(
        ChatRequest(
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
    )
    assert list(stream) == ["secondary"]
    assert len(primary.requests) == 1
    assert len(secondary.requests) == 1


def test_async_fallback_uses_the_same_error_policy() -> None:
    async def run() -> tuple[str, int]:
        registry = ProviderRegistry()
        primary = AsyncScriptedChatClient(
            [VvLlmError("temporary", kind=ErrorKind.PROVIDER_INTERNAL)]
        )
        secondary = AsyncScriptedChatClient(["ok"])
        capabilities = ModelCapabilities()
        registry.register("primary", lambda: primary, capabilities=capabilities)
        registry.register("secondary", lambda: secondary, capabilities=capabilities)
        client = AsyncFallbackChatClient(
            registry,
            [
                FallbackRoute("primary", "model-a"),
                FallbackRoute("secondary", "model-b"),
            ],
        )
        result = await client.create_with_metadata(
            ChatRequest(messages=[{"role": "user", "content": "hello"}])
        )
        return result.response, result.metadata.fallback_index

    assert asyncio.run(run()) == ("ok", 1)
