from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from vv_llm import CapabilityPolicy, ChatRequest
from vv_llm.chat_clients.deepseek_client import DeepSeekChatClient
from vv_llm.contract import load_fixture
from vv_llm.settings import Settings
from vv_llm.types.exception import _retry_after


def _fixture() -> dict[str, Any]:
    return load_fixture("openai-compatible.v2.json")


def _retry_fixture() -> dict[str, Any]:
    return load_fixture("retry-after.v1.json")


def _settings(model: str) -> Settings:
    return Settings.load_from_dict(
        {
            "VERSION": "2",
            "rate_limit": {"enabled": False},
            "endpoints": [
                {
                    "id": "fixture-endpoint",
                    "api_base": "https://example.invalid/v1",
                    "api_key": "test-key",
                }
            ],
            "backends": {
                "deepseek": {
                    "models": {
                        model: {
                            "id": model,
                            "endpoints": ["fixture-endpoint"],
                            "context_length": 8192,
                            "max_output_tokens": 1024,
                            "function_call_available": True,
                        }
                    }
                }
            },
        }
    )


def _request_from_fixture(value: dict[str, Any], *, stream: bool | None = None) -> ChatRequest:
    request = ChatRequest.from_contract(value)
    if stream is not None:
        request = request.model_copy(update={"stream": stream})
    return request


def _bind_raw_client(client: DeepSeekChatClient, create: Any) -> None:
    client.endpoint = client.settings.get_endpoint("fixture-endpoint")
    client.model_id = client.model
    client.raw_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _usage(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    cached_tokens = None
    if value.prompt_tokens_details is not None:
        cached_tokens = value.prompt_tokens_details.cached_tokens
    return {
        "prompt_tokens": value.prompt_tokens,
        "completion_tokens": value.completion_tokens,
        "total_tokens": value.total_tokens,
        "cache_read_input_tokens": cached_tokens,
        "cache_creation_input_tokens": value.cache_creation_tokens,
    }


def _tool_calls(value: Any) -> list[dict[str, Any]]:
    calls = []
    for tool_call in value or []:
        function = tool_call.function
        item = {
            "id": tool_call.id or "",
            "name": function.name or "",
            "arguments": function.arguments or "",
        }
        if getattr(tool_call, "index", None) is not None:
            item["index"] = tool_call.index
        calls.append(item)
    return calls


def test_openai_compatible_fixture_covers_request_completion_and_stream() -> None:
    fixture = _fixture()
    canonical = fixture["request_case"]["canonical_request"]
    settings = _settings(canonical["model"])
    captured: dict[str, Any] = {}

    def create_completion(**kwargs: Any) -> ChatCompletion:
        captured.update(kwargs)
        return ChatCompletion.model_validate(fixture["completion_case"]["raw_response"])

    client = DeepSeekChatClient(model=canonical["model"], stream=False, settings=settings)
    _bind_raw_client(client, create_completion)
    request = _request_from_fixture(canonical)
    canonical_roundtrip = {
        **canonical,
        "messages": [dict(message) for message in fixture["request_case"]["expected_wire_request"]["messages"]],
    }
    canonical_roundtrip["messages"][1] = {
        **canonical_roundtrip["messages"][1],
        "tool_calls": canonical["messages"][1]["tool_calls"],
    }
    assert request.to_contract() == canonical_roundtrip
    result = client.create(
        request,
        capability_policy=CapabilityPolicy.PASSTHROUGH,
    )

    expected_request = fixture["request_case"]["expected_wire_request"]
    flattened_request = {key: captured[key] for key in expected_request if key in captured and key not in canonical["extra_body"] and key != "thinking"}
    flattened_request.update(captured["extra_body"])
    assert flattened_request == expected_request

    assert {
        "content": result.content,
        "reasoning_content": result.reasoning_content,
        "tool_calls": _tool_calls(result.tool_calls),
        "usage": _usage(result.usage),
    } == fixture["completion_case"]["expected_response"]

    chunks = [ChatCompletionChunk.model_validate(value) for value in fixture["stream_case"]["raw_chunks"]]
    stream_client = DeepSeekChatClient(model=canonical["model"], stream=True, settings=settings)
    _bind_raw_client(stream_client, lambda **kwargs: iter(chunks))
    actual_deltas = []
    for delta in stream_client.create(
        _request_from_fixture(canonical, stream=True),
        capability_policy=CapabilityPolicy.PASSTHROUGH,
    ):
        item: dict[str, Any] = {
            "content": delta.content or "",
            "reasoning_content": delta.reasoning_content or "",
            "tool_calls": _tool_calls(delta.tool_calls),
        }
        if delta.usage is not None:
            item["usage"] = _usage(delta.usage)
        actual_deltas.append(item)

    assert actual_deltas == fixture["stream_case"]["expected_deltas"]


def test_retry_after_fixture_cases() -> None:
    for case in _retry_fixture()["cases"]:
        response = SimpleNamespace(headers=case["headers"])
        now = datetime.fromtimestamp(case["now_unix_seconds"], tz=timezone.utc)
        assert _retry_after(response, now=now) == case["expected_seconds"], case["name"]
