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
            "VERSION": "2",
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
