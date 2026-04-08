from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from vv_llm.chat_clients.gemini_client import GeminiChatClient
from vv_llm.settings import Settings


SETTINGS_PAYLOAD = {
    "VERSION": "2",
    "endpoints": [
        {
            "id": "gemini-test",
            "api_base": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key": "test-key",
            "enabled": True,
        }
    ],
    "backends": {
        "gemini": {
            "models": {
                "gemini-3.1-pro-preview": {
                    "id": "gemini-3.1-pro-preview",
                    "endpoints": ["gemini-test"],
                    "function_call_available": True,
                    "response_format_available": True,
                    "native_multimodal": True,
                    "context_length": 32768,
                    "max_output_tokens": 8192,
                }
            }
        }
    },
}


class FakeToolFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments

    def model_dump(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str, extra_content: dict[str, Any] | None = None) -> None:
        self.id = call_id
        self.function = FakeToolFunction(name=name, arguments=arguments)
        self.extra_content = extra_content
        self.index: int | None = None
        self.type: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "index": self.index,
            "type": self.type,
            "function": self.function.model_dump(),
        }


class FakeDelta:
    def __init__(
        self,
        *,
        content: str | None = None,
        reasoning_content: str | None = None,
        tool_calls: list[FakeToolCall] | None = None,
        extra_content: dict[str, Any] | None = None,
    ) -> None:
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls
        self.extra_content = extra_content

    def model_dump(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.content is not None:
            payload["content"] = self.content
        if self.reasoning_content is not None:
            payload["reasoning_content"] = self.reasoning_content
        if self.tool_calls is not None:
            payload["tool_calls"] = [tool_call.model_dump() for tool_call in self.tool_calls]
        return payload


def _make_stream_chunk(delta: FakeDelta) -> SimpleNamespace:
    return SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(delta=delta)],
    )


def _make_client() -> GeminiChatClient:
    settings = Settings.load_from_dict(SETTINGS_PAYLOAD)
    client = GeminiChatClient(
        model="gemini-3.1-pro-preview",
        stream=True,
        endpoint_id="gemini-test",
        settings=settings,
    )
    client.endpoint, client.model_id = client._set_endpoint()
    return client


def test_gemini_stream_strips_thought_tags_when_tool_call_arrives() -> None:
    client = _make_client()
    stream_chunks = [
        _make_stream_chunk(FakeDelta(reasoning_content="<thought>Initiating Image Transcription\n\n")),
        _make_stream_chunk(
            FakeDelta(
                content="</thought>",
                tool_calls=[
                    FakeToolCall(
                        call_id="ip8bq3e7",
                        name="_read_image",
                        arguments='{"path":"image.jpg"}',
                        extra_content={"google": {"thought_signature": "abc"}},
                    )
                ],
            )
        ),
    ]
    client.__dict__["raw_client"] = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: iter(stream_chunks),
            )
        )
    )

    response = list(
        client.create_stream(
            messages=[{"role": "user", "content": "Describe the image"}],
            skip_cutoff=True,
            max_tokens=16,
        )
    )

    assert len(response) == 2
    assert response[0].content == ""
    assert response[0].reasoning_content == "Initiating Image Transcription\n\n"
    assert response[1].content == ""
    assert response[1].tool_calls is not None
    assert response[1].tool_calls[0].function.name == "_read_image"
    assert response[1].raw_content == {"google": [{"thought_signature": "abc"}]}


def test_gemini_stream_strips_thought_tags_from_google_thought_chunks() -> None:
    client = _make_client()
    stream_chunks = [
        _make_stream_chunk(
            FakeDelta(
                content="<thought>Initiating Image Transcription\n\n",
                extra_content={"google": {"thought": True}},
            )
        ),
        _make_stream_chunk(
            FakeDelta(
                content="</thought>",
                tool_calls=[
                    FakeToolCall(
                        call_id="ip8bq3e7",
                        name="_read_image",
                        arguments='{"path":"image.jpg"}',
                        extra_content={"google": {"thought_signature": "abc"}},
                    )
                ],
            )
        ),
    ]
    client.__dict__["raw_client"] = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: iter(stream_chunks),
            )
        )
    )

    response = list(
        client.create_stream(
            messages=[{"role": "user", "content": "Describe the image"}],
            skip_cutoff=True,
            max_tokens=16,
        )
    )

    assert len(response) == 2
    assert response[0].reasoning_content == "Initiating Image Transcription\n\n"
    assert response[0].content == ""
    assert response[1].content == ""
    assert response[1].tool_calls is not None
    assert response[1].tool_calls[0].function.name == "_read_image"


def test_gemini_non_stream_strips_thought_tags_from_reasoning_content() -> None:
    client = _make_client()
    response_message = SimpleNamespace(
        content="",
        reasoning_content="<thought>Reasoned answer</thought>",
        tool_calls=None,
    )
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=response_message)],
        usage=None,
    )
    client.__dict__["raw_client"] = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: fake_response,
            )
        )
    )

    response = client.create_completion(
        messages=[{"role": "user", "content": "Say hi"}],
        stream=False,
        skip_cutoff=True,
        max_tokens=16,
    )

    assert response.reasoning_content == "Reasoned answer"
    assert response.content == ""
