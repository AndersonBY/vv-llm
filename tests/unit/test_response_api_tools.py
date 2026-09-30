"""Regression tests for the Responses API tool-call adapter."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from vv_llm.chat_clients.message_normalizer import messages_for_response_api
from vv_llm.chat_clients.openai_client import AsyncOpenAIChatClient, OpenAIChatClient
from vv_llm.chat_clients.stream_event_adapter import adapt_response_api_stream_event
from vv_llm.settings import Settings


ENDPOINT_ID = "responses-test"
MODEL = "gpt-6-astra"
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]


def _settings() -> Settings:
    return Settings.load_from_dict(
        {
            "rate_limit": {"enabled": False},
            "endpoints": [
                {
                    "id": ENDPOINT_ID,
                    "api_base": "https://example.invalid/v1",
                    "api_key": "test-key",
                    "response_api": True,
                }
            ],
            "backends": {
                "openai": {
                    "models": {
                        MODEL: {
                            "id": MODEL,
                            "endpoints": [ENDPOINT_ID],
                            "context_length": 100000,
                            "max_output_tokens": 1024,
                        }
                    }
                }
            },
        }
    )


def _bind_raw_client(client: Any, responses: Any) -> None:
    client.endpoint = client.settings.get_endpoint(ENDPOINT_ID)
    client.model_id = MODEL
    client.__dict__["raw_client"] = SimpleNamespace(responses=responses)


def _function_call_item(name: str, call_id: str, item_id: str, arguments: str = "") -> SimpleNamespace:
    return SimpleNamespace(type="function_call", id=item_id, call_id=call_id, name=name, arguments=arguments)


def _event(event_type: str, **kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, **kwargs)


def _interleaved_tool_events() -> list[SimpleNamespace]:
    """Two parallel tool calls whose deltas and terminal events interleave."""

    return [
        _event("response.output_item.added", output_index=0, item=_function_call_item("get_weather", "call_a", "fc_1")),
        _event("response.output_item.added", output_index=1, item=_function_call_item("get_time", "call_b", "fc_2")),
        _event("response.function_call_arguments.delta", output_index=0, delta='{"city"'),
        _event("response.function_call_arguments.delta", output_index=1, delta='{"tz"'),
        _event("response.function_call_arguments.delta", output_index=0, delta=':"SF"}'),
        _event("response.function_call_arguments.done", output_index=0, arguments='{"city":"SF"}'),
        _event("response.function_call_arguments.delta", output_index=1, delta=':"UTC"}'),
        _event("response.output_item.done", output_index=0, item=_function_call_item("get_weather", "call_a", "fc_1", '{"city":"SF"}')),
        _event("response.output_item.done", output_index=1, item=_function_call_item("get_time", "call_b", "fc_2", '{"tz":"UTC"}')),
        _event("response.completed", response=SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=20))),
    ]


def _accumulate(deltas: list[Any]) -> tuple[dict[int, dict[str, Any]], bool]:
    accumulated: dict[int, dict[str, Any]] = {}
    usage_seen = False
    for message in deltas:
        if message.usage is not None:
            usage_seen = True
        for call in message.tool_calls or []:
            entry = accumulated.setdefault(call.index, {"id": None, "name": None, "arguments": ""})
            if call.id:
                entry["id"] = call.id
            if call.function and call.function.name:
                entry["name"] = call.function.name
            if call.function and call.function.arguments:
                entry["arguments"] += call.function.arguments
    return accumulated, usage_seen


class _SyncStreamContext:
    def __init__(self, events: list[Any]) -> None:
        self._events = events

    def __enter__(self) -> Any:
        return iter(self._events)

    def __exit__(self, *_: Any) -> bool:
        return False


class _AsyncStreamContext:
    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def __aenter__(self) -> _AsyncStreamContext:
        return self

    def __aiter__(self) -> Any:
        return self._iterate()

    async def _iterate(self) -> Any:
        for event in self._events:
            yield event

    async def __aexit__(self, *_: Any) -> bool:
        return False


class _FakeResponses:
    def __init__(self, *, result: Any = None, events: list[Any] | None = None) -> None:
        self._result = result
        self._events = events or []
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._result

    def stream(self, **kwargs: Any) -> _SyncStreamContext:
        self.calls.append(kwargs)
        return _SyncStreamContext(self._events)


class _FakeAsyncResponses(_FakeResponses):
    async def create(self, **kwargs: Any) -> Any:  # type: ignore[override]
        self.calls.append(kwargs)
        return self._result

    def stream(self, **kwargs: Any) -> _AsyncStreamContext:  # type: ignore[override]
        self.calls.append(kwargs)
        return _AsyncStreamContext(self._events)


def test_messages_for_response_api_rebuilds_tool_history() -> None:
    converted = messages_for_response_api(
        [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": '{"temp": 21}'},
        ]
    )

    assert converted == [
        {"role": "user", "content": "weather?"},
        {
            "type": "function_call",
            "call_id": "call_a",
            "name": "get_weather",
            "arguments": '{"city": "SF"}',
        },
        {"type": "function_call_output", "call_id": "call_a", "output": '{"temp": 21}'},
    ]


def test_messages_for_response_api_converts_content_parts() -> None:
    converted = messages_for_response_api(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}},
                ],
            }
        ]
    )

    assert converted == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "describe"},
                {"type": "input_image", "image_url": "https://example.invalid/a.png"},
            ],
        }
    ]


def test_sync_two_turn_tool_call_uses_call_id_and_response_items() -> None:
    first_turn = SimpleNamespace(
        output_text="",
        output=[_function_call_item("get_weather", "call_a", "fc_1", '{"city": "SF"}')],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )
    responses = _FakeResponses(result=first_turn)
    client = OpenAIChatClient(model=MODEL, stream=False, settings=_settings())
    _bind_raw_client(client, responses)

    result = client.create_completion(
        messages=[{"role": "user", "content": "weather?"}],
        tools=TOOLS,
        stream=False,
        skip_cutoff=True,
    )

    assert [call.id for call in result.tool_calls or []] == ["call_a"]
    assert json.loads(result.tool_calls[0].function.arguments) == {"city": "SF"}

    responses._result = SimpleNamespace(
        output_text="It is sunny.",
        output=[],
        usage=SimpleNamespace(input_tokens=20, output_tokens=5),
    )
    client.create_completion(
        messages=[
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": '{"temp": 21}'},
        ],
        tools=TOOLS,
        stream=False,
        skip_cutoff=True,
    )

    second_input = responses.calls[-1]["input"]
    assert second_input == [
        {"role": "user", "content": "weather?"},
        {
            "type": "function_call",
            "call_id": "call_a",
            "name": "get_weather",
            "arguments": '{"city": "SF"}',
        },
        {"type": "function_call_output", "call_id": "call_a", "output": '{"temp": 21}'},
    ]


def test_async_two_turn_tool_call_uses_call_id_and_response_items() -> None:
    first_turn = SimpleNamespace(
        output_text="",
        output=[_function_call_item("get_weather", "call_a", "fc_1", '{"city": "SF"}')],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )
    responses = _FakeAsyncResponses(result=first_turn)

    async def run() -> Any:
        client = AsyncOpenAIChatClient(model=MODEL, stream=False, settings=_settings())
        _bind_raw_client(client, responses)
        result = await client.create_completion(
            messages=[{"role": "user", "content": "weather?"}],
            tools=TOOLS,
            stream=False,
            skip_cutoff=True,
        )
        responses._result = SimpleNamespace(
            output_text="It is sunny.",
            output=[],
            usage=SimpleNamespace(input_tokens=20, output_tokens=5),
        )
        await client.create_completion(
            messages=[
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_a", "content": '{"temp": 21}'},
            ],
            tools=TOOLS,
            stream=False,
            skip_cutoff=True,
        )
        return result

    result = asyncio.run(run())

    assert [call.id for call in result.tool_calls or []] == ["call_a"]
    assert responses.calls[-1]["input"][1]["call_id"] == "call_a"
    assert responses.calls[-1]["input"][2]["type"] == "function_call_output"


def test_interleaved_stream_events_emit_each_argument_once() -> None:
    final_tool_calls: dict[int, dict[str, Any]] = {}
    deltas: list[Any] = []
    usage = None
    for event in _interleaved_tool_events():
        messages, usage_from_event = adapt_response_api_stream_event(event, final_tool_calls, is_gemini3=False)
        usage = usage_from_event or usage
        deltas.extend(messages)

    accumulated, _ = _accumulate(deltas)

    assert accumulated == {
        0: {"id": "call_a", "name": "get_weather", "arguments": '{"city":"SF"}'},
        1: {"id": "call_b", "name": "get_time", "arguments": '{"tz":"UTC"}'},
    }
    assert usage is not None and usage.total_tokens == 30


def test_stream_terminal_event_replays_arguments_when_no_deltas_arrived() -> None:
    final_tool_calls: dict[int, dict[str, Any]] = {}
    messages, _ = adapt_response_api_stream_event(
        _event("response.output_item.done", output_index=0, item=_function_call_item("get_weather", "call_a", "fc_1", '{"city":"SF"}')),
        final_tool_calls,
        is_gemini3=False,
    )

    accumulated, _ = _accumulate(messages)
    assert accumulated == {0: {"id": "call_a", "name": "get_weather", "arguments": '{"city":"SF"}'}}


def test_sync_responses_stream_accumulates_interleaved_tool_calls() -> None:
    responses = _FakeResponses(events=_interleaved_tool_events())
    client = OpenAIChatClient(model=MODEL, stream=True, settings=_settings())
    _bind_raw_client(client, responses)

    deltas = list(client.create_completion(messages=[{"role": "user", "content": "weather?"}], tools=TOOLS, stream=True, skip_cutoff=True))
    accumulated, usage_seen = _accumulate(deltas)

    assert accumulated == {
        0: {"id": "call_a", "name": "get_weather", "arguments": '{"city":"SF"}'},
        1: {"id": "call_b", "name": "get_time", "arguments": '{"tz":"UTC"}'},
    }
    assert usage_seen


def test_async_responses_stream_accumulates_interleaved_tool_calls() -> None:
    responses = _FakeAsyncResponses(events=_interleaved_tool_events())

    async def run() -> list[Any]:
        client = AsyncOpenAIChatClient(model=MODEL, stream=True, settings=_settings())
        _bind_raw_client(client, responses)
        deltas = []
        async for message in await client.create_completion(messages=[{"role": "user", "content": "weather?"}], tools=TOOLS, stream=True, skip_cutoff=True):
            deltas.append(message)
        return deltas

    accumulated, usage_seen = _accumulate(asyncio.run(run()))

    assert accumulated == {
        0: {"id": "call_a", "name": "get_weather", "arguments": '{"city":"SF"}'},
        1: {"id": "call_b", "name": "get_time", "arguments": '{"tz":"UTC"}'},
    }
    assert usage_seen
