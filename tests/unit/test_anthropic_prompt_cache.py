from __future__ import annotations

import asyncio
from types import SimpleNamespace

from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message, RawMessageDeltaEvent, RawMessageStartEvent, TextBlock, Usage as AnthropicUsage
from anthropic.types.raw_message_delta_event import Delta, MessageDeltaUsage

from vv_llm.chat_clients.anthropic_client import AnthropicChatClient, AsyncAnthropicChatClient
from vv_llm.chat_clients.stream_event_adapter import adapt_anthropic_stream_event, build_anthropic_completion_message, init_anthropic_stream_state
from vv_llm.settings import Settings


TEST_MODEL = "claude-sonnet-4-5-20250929"


def _anthropic_settings() -> Settings:
    return Settings.load_from_dict(
        {
            "rate_limit": {"enabled": False},
            "endpoints": [
                {
                    "id": "anthropic-test",
                    "api_base": "https://api.anthropic.com",
                    "api_key": "test-key",
                    "endpoint_type": "anthropic",
                }
            ],
            "backends": {
                "anthropic": {
                    "models": {
                        TEST_MODEL: {
                            "id": TEST_MODEL,
                            "endpoints": ["anthropic-test"],
                            "context_length": 8192,
                            "max_output_tokens": 1024,
                        }
                    }
                }
            },
        }
    )


class _FakeAnthropic(Anthropic):
    def __init__(self, create_impl):
        self.messages = SimpleNamespace(create=create_impl)


class _FakeAsyncAnthropic(AsyncAnthropic):
    def __init__(self, create_impl):
        self.messages = SimpleNamespace(create=create_impl)


def test_build_anthropic_completion_message_tracks_cache_creation_tokens() -> None:
    message = build_anthropic_completion_message(
        [TextBlock(type="text", text="hello")],
        AnthropicUsage(
            input_tokens=10,
            output_tokens=4,
            cache_read_input_tokens=3,
            cache_creation_input_tokens=7,
        ),
    )

    assert message.content == "hello"
    assert message.usage is not None
    assert message.usage.prompt_tokens == 20
    assert message.usage.completion_tokens == 4
    assert message.usage.total_tokens == 24
    assert message.usage.cache_creation_tokens == 7
    assert message.usage.prompt_tokens_details is not None
    assert message.usage.prompt_tokens_details.cached_tokens == 3


def test_adapt_anthropic_stream_event_tracks_cache_creation_tokens() -> None:
    stream_state = init_anthropic_stream_state()

    start_event = RawMessageStartEvent(
        type="message_start",
        message=Message(
            id="msg_1",
            content=[],
            model=TEST_MODEL,
            role="assistant",
            stop_reason=None,
            stop_sequence=None,
            type="message",
            usage=AnthropicUsage(
                input_tokens=10,
                output_tokens=0,
                cache_read_input_tokens=3,
                cache_creation_input_tokens=7,
            ),
        ),
    )
    assert adapt_anthropic_stream_event(start_event, stream_state) is None

    delta_event = RawMessageDeltaEvent(
        type="message_delta",
        delta=Delta(stop_reason=None, stop_sequence=None),
        usage=MessageDeltaUsage(output_tokens=4, input_tokens=10, cache_creation_input_tokens=7, cache_read_input_tokens=3),
    )
    message = adapt_anthropic_stream_event(delta_event, stream_state)

    assert message is not None
    assert message.usage is not None
    assert message.usage.prompt_tokens == 20
    assert message.usage.completion_tokens == 4
    assert message.usage.total_tokens == 24
    assert message.usage.cache_creation_tokens == 7
    assert message.usage.prompt_tokens_details is not None
    assert message.usage.prompt_tokens_details.cached_tokens == 3


def test_anthropic_client_forwards_extra_body_request_options() -> None:
    client = AnthropicChatClient(
        model=TEST_MODEL,
        endpoint_id="anthropic-test",
        random_endpoint=False,
        settings=_anthropic_settings(),
    )
    captured: dict[str, object] = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(content=[], usage=AnthropicUsage(input_tokens=2, output_tokens=1))

    client._cached_raw_client = _FakeAnthropic(fake_create)
    client._acquire_rate_limit = lambda *args, **kwargs: None

    response = client.create_completion(
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        max_tokens=32,
        extra_query={"trace": "1"},
        extra_body={"cache_control": {"type": "ephemeral"}},
        timeout=12,
    )

    assert response.usage is not None
    assert captured["extra_query"] == {"trace": "1"}
    assert captured["extra_body"] == {"cache_control": {"type": "ephemeral"}}
    assert captured["timeout"] == 12


def test_async_anthropic_client_forwards_extra_body_request_options() -> None:
    async def run() -> dict[str, object]:
        client = AsyncAnthropicChatClient(
            model=TEST_MODEL,
            endpoint_id="anthropic-test",
            random_endpoint=False,
            settings=_anthropic_settings(),
        )
        captured: dict[str, object] = {}

        async def fake_create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[], usage=AnthropicUsage(input_tokens=2, output_tokens=1))

        async def fake_rate_limit(*args, **kwargs):
            return None

        client._cached_raw_client = _FakeAsyncAnthropic(fake_create)
        client._acquire_rate_limit = fake_rate_limit

        response = await client.create_completion(
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
            max_tokens=32,
            extra_query={"trace": "1"},
            extra_body={"cache_control": {"type": "ephemeral"}},
            timeout=12,
        )

        assert response.usage is not None
        return captured

    captured = asyncio.run(run())

    assert captured["extra_query"] == {"trace": "1"}
    assert captured["extra_body"] == {"cache_control": {"type": "ephemeral"}}
    assert captured["timeout"] == 12


def test_anthropic_sampling_parameters_are_nested_in_extra_body_without_overwrite() -> None:
    calls: list[dict[str, object]] = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=[], usage=AnthropicUsage(input_tokens=2, output_tokens=1))

    client = AnthropicChatClient(
        model=TEST_MODEL,
        endpoint_id="anthropic-test",
        random_endpoint=False,
        settings=_anthropic_settings(),
    )
    client._cached_raw_client = _FakeAnthropic(fake_create)
    client._acquire_rate_limit = lambda *args, **kwargs: None

    client.create_completion(
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        max_tokens=32,
        temperature=0.2,
        top_p=0.7,
        extra_body={"trace": "generated"},
    )
    generated = calls[-1]
    assert "temperature" not in generated
    assert "top_p" not in generated
    assert generated["extra_body"] == {"trace": "generated", "temperature": 0.2, "top_p": 0.7}

    client = AnthropicChatClient(
        model=TEST_MODEL,
        endpoint_id="anthropic-test",
        random_endpoint=False,
        settings=_anthropic_settings(),
    )
    client._cached_raw_client = _FakeAnthropic(fake_create)
    client._acquire_rate_limit = lambda *args, **kwargs: None
    client.create_completion(
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        max_tokens=32,
        temperature=0.2,
        top_p=0.7,
        extra_body={"temperature": 0.9, "top_p": 0.8, "trace": "user"},
    )
    preserved = calls[-1]
    assert "temperature" not in preserved
    assert "top_p" not in preserved
    assert preserved["extra_body"] == {"temperature": 0.9, "top_p": 0.8, "trace": "user"}


def test_async_anthropic_sampling_parameters_are_nested_in_extra_body_without_overwrite() -> None:
    async def run() -> list[dict[str, object]]:
        calls: list[dict[str, object]] = []

        async def fake_create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(content=[], usage=AnthropicUsage(input_tokens=2, output_tokens=1))

        client = AsyncAnthropicChatClient(
            model=TEST_MODEL,
            endpoint_id="anthropic-test",
            random_endpoint=False,
            settings=_anthropic_settings(),
        )
        client._cached_raw_client = _FakeAsyncAnthropic(fake_create)

        async def fake_rate_limit(*args, **kwargs):
            return None

        client._acquire_rate_limit = fake_rate_limit
        await client.create_completion(
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
            max_tokens=32,
            temperature=0.2,
            top_p=0.7,
            extra_body={"trace": "generated"},
        )

        client = AsyncAnthropicChatClient(
            model=TEST_MODEL,
            endpoint_id="anthropic-test",
            random_endpoint=False,
            settings=_anthropic_settings(),
        )
        client._cached_raw_client = _FakeAsyncAnthropic(fake_create)
        client._acquire_rate_limit = fake_rate_limit
        await client.create_completion(
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
            max_tokens=32,
            temperature=0.2,
            top_p=0.7,
            extra_body={"temperature": 0.9, "top_p": 0.8, "trace": "user"},
        )
        return calls

    generated, preserved = asyncio.run(run())
    assert "temperature" not in generated
    assert "top_p" not in generated
    assert generated["extra_body"] == {"trace": "generated", "temperature": 0.2, "top_p": 0.7}
    assert "temperature" not in preserved
    assert "top_p" not in preserved
    assert preserved["extra_body"] == {"temperature": 0.9, "top_p": 0.8, "trace": "user"}


def _tool_stream_events():
    from anthropic.types import RawContentBlockDeltaEvent, RawContentBlockStartEvent

    def start(index, call_id, name):
        return RawContentBlockStartEvent(type="content_block_start", index=index, content_block={"type": "tool_use", "id": call_id, "name": name, "input": {}})

    def delta(index, text):
        return RawContentBlockDeltaEvent(type="content_block_delta", index=index, delta={"type": "input_json_delta", "partial_json": text})

    return [
        RawContentBlockStartEvent(type="content_block_start", index=3, content_block={"type": "text", "text": "preface"}),
        start(7, "call_a", "f"),
        RawContentBlockStartEvent(type="content_block_start", index=5, content_block={"type": "thinking", "thinking": "plan", "signature": "sig"}),
        start(0, "call_b", "g"),
        delta(7, '{"x":'),
        delta(0, '{"y":'),
        delta(7, "1}"),
        delta(0, "2}"),
    ]


def _assert_tool_stream(messages):
    calls = [call for message in messages if message is not None for call in message.tool_calls or []]
    assert [call.index for call in calls] == [7, 0, 7, 0, 7, 0]
    assert [call.id for call in calls] == ["call_a", "call_b", "call_a", "call_b", "call_a", "call_b"]
    assert [call.function.name for call in calls] == ["f", "g", "f", "g", "f", "g"]
    assert [call.function.arguments for call in calls] == ["", "", '{"x":', '{"y":', "1}", "2}"]


def test_anthropic_adapter_tracks_interleaved_tool_blocks_by_provider_index():
    state = init_anthropic_stream_state()
    messages = [adapt_anthropic_stream_event(event, state) for event in _tool_stream_events()]
    _assert_tool_stream(messages)
    assert [block["input"] for block in state["raw_content"] if block["type"] == "tool_use"] == [{"x": 1}, {"y": 2}]


def test_anthropic_adapter_does_not_attach_unknown_block_delta_to_last_tool():
    from anthropic.types import RawContentBlockDeltaEvent

    state = init_anthropic_stream_state()
    adapt_anthropic_stream_event(_tool_stream_events()[1], state)
    message = adapt_anthropic_stream_event(RawContentBlockDeltaEvent(type="content_block_delta", index=99, delta={"type": "input_json_delta", "partial_json": "{}"}), state)
    assert not message.tool_calls
    assert state["raw_content"][0]["input"] == {}
    # A malformed missing/null index must not guess ownership from the last call.
    anonymous = RawContentBlockDeltaEvent.construct(type="content_block_delta", index=None, delta={"type": "input_json_delta", "partial_json": "{}"})
    assert not adapt_anthropic_stream_event(anonymous, state).tool_calls


def _reject_real_sdk_client(*args, **kwargs):
    raise AssertionError("offline streaming test must not construct a real SDK client")


def test_sync_anthropic_client_preserves_interleaved_tool_identity(monkeypatch):
    monkeypatch.setattr(Anthropic, "__init__", _reject_real_sdk_client)
    client = AnthropicChatClient(model=TEST_MODEL, endpoint_id="anthropic-test", stream=True, settings=_anthropic_settings())
    client._cached_raw_client = _FakeAnthropic(lambda **kwargs: iter(_tool_stream_events()))
    _assert_tool_stream(list(client.create_completion(messages=[{"role": "user", "content": "hello"}], stream=True, max_tokens=16, skip_cutoff=True)))


def test_async_anthropic_client_preserves_interleaved_tool_identity(monkeypatch):
    monkeypatch.setattr(AsyncAnthropic, "__init__", _reject_real_sdk_client)

    async def run():
        async def chunks():
            for event in _tool_stream_events():
                yield event

        async def create(**kwargs):
            return chunks()

        client = AsyncAnthropicChatClient(model=TEST_MODEL, endpoint_id="anthropic-test", stream=True, settings=_anthropic_settings())
        client._cached_raw_client = _FakeAsyncAnthropic(create)
        response = await client.create_completion(messages=[{"role": "user", "content": "hello"}], stream=True, max_tokens=16, skip_cutoff=True)
        _assert_tool_stream([message async for message in response])

    asyncio.run(run())
