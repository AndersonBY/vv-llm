from __future__ import annotations

import asyncio
from types import SimpleNamespace

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


def test_stream_replays_usage_done_and_empty_prelude_once() -> None:
    registry = ProviderRegistry()
    usage = SimpleNamespace(usage={"total_tokens": 1})
    done = SimpleNamespace(done=True)
    empty = SimpleNamespace()
    tool = SimpleNamespace(tool_calls=[{"id": "call-1"}])
    primary = ScriptedChatClient(
        [
            ScriptedStream(
                [
                    usage,
                    done,
                    empty,
                    tool,
                    "tail",
                ]
            )
        ]
    )
    registry.register(
        "primary",
        lambda: primary,
        capabilities=ModelCapabilities(streaming=True),
    )
    client = FallbackChatClient(registry, [FallbackRoute("primary", "model-a")])

    stream = client.create(
        ChatRequest(messages=[{"role": "user", "content": "hello"}], stream=True)
    )

    chunks = list(stream)
    assert chunks[:3] == [usage, done, empty]
    assert chunks[3] is tool
    assert chunks[4:] == ["tail"]


def test_stream_replays_nonvisible_prelude_when_no_output_arrives() -> None:
    registry = ProviderRegistry()
    usage = SimpleNamespace(usage={"total_tokens": 1})
    done = SimpleNamespace(done=True)
    empty = SimpleNamespace()
    primary = ScriptedChatClient(
        [
            ScriptedStream(
                [
                    usage,
                    done,
                    empty,
                ]
            )
        ]
    )
    registry.register(
        "primary",
        lambda: primary,
        capabilities=ModelCapabilities(streaming=True),
    )
    client = FallbackChatClient(registry, [FallbackRoute("primary", "model-a")])

    stream = client.create(
        ChatRequest(messages=[{"role": "user", "content": "hello"}], stream=True)
    )

    assert list(stream) == [usage, done, empty]


def test_stream_falls_back_after_nonvisible_prelude_before_error() -> None:
    registry = ProviderRegistry()
    primary = ScriptedChatClient(
        [
            ScriptedStream(
                [
                    SimpleNamespace(usage={"total_tokens": 1}),
                    SimpleNamespace(done=True),
                    VvLlmError("temporary", kind=ErrorKind.NETWORK),
                ]
            )
        ]
    )
    secondary_empty = SimpleNamespace()
    secondary = ScriptedChatClient(
        [ScriptedStream([secondary_empty, "secondary-visible"])]
    )
    capabilities = ModelCapabilities(streaming=True)
    registry.register("primary", lambda: primary, capabilities=capabilities)
    registry.register("secondary", lambda: secondary, capabilities=capabilities)
    client = FallbackChatClient(
        registry,
        [FallbackRoute("primary", "model-a"), FallbackRoute("secondary", "model-b")],
    )

    stream = client.create(
        ChatRequest(messages=[{"role": "user", "content": "hello"}], stream=True)
    )

    chunks = list(stream)
    assert chunks[0] is secondary_empty
    assert chunks[1:] == ["secondary-visible"]


def test_async_stream_falls_back_after_nonvisible_prelude_before_error() -> None:
    async def run() -> tuple[list[object], tuple[object, ...], tuple[object, ...], object]:
        registry = ProviderRegistry()
        primary_usage = SimpleNamespace(usage={"total_tokens": 1})
        primary_done = SimpleNamespace(done=True)
        secondary_empty = SimpleNamespace()
        primary = AsyncScriptedChatClient(
            [
                ScriptedStream(
                    [
                        primary_usage,
                        primary_done,
                        VvLlmError("temporary", kind=ErrorKind.NETWORK),
                    ]
                )
            ]
        )
        secondary = AsyncScriptedChatClient(
            [ScriptedStream([secondary_empty, SimpleNamespace(reasoning_content="reasoning")])]
        )
        capabilities = ModelCapabilities(streaming=True)
        registry.register("primary", lambda: primary, capabilities=capabilities)
        registry.register("secondary", lambda: secondary, capabilities=capabilities)
        client = AsyncFallbackChatClient(
            registry,
            [FallbackRoute("primary", "model-a"), FallbackRoute("secondary", "model-b")],
        )
        stream = await client.create(
            ChatRequest(messages=[{"role": "user", "content": "hello"}], stream=True)
        )
        return [chunk async for chunk in stream], primary.requests, secondary.requests, secondary_empty

    chunks, primary_requests, secondary_requests, secondary_empty = asyncio.run(run())
    assert chunks[0] is secondary_empty
    assert getattr(chunks[0], "reasoning_content", None) is None
    assert getattr(chunks[0], "usage", None) is None
    assert chunks[1].reasoning_content == "reasoning"
    assert len(primary_requests) == 1
    assert len(secondary_requests) == 1


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
