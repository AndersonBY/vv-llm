from __future__ import annotations

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from openai._types import NOT_GIVEN as OPENAI_NOT_GIVEN

from vv_llm.chat_clients.openai_client import AsyncOpenAIChatClient, OpenAIChatClient
from vv_llm.chat_clients.openai_compatible_client import _uses_max_completion_tokens
from vv_llm.settings import Settings


ENDPOINT_ID = "openai-compatible-test"


def _settings(model: str) -> Settings:
    return Settings.load_from_dict(
        {
            "rate_limit": {"enabled": False},
            "endpoints": [
                {
                    "id": ENDPOINT_ID,
                    "api_base": "https://example.invalid/v1",
                    "api_key": "test-key",
                }
            ],
            "backends": {
                "openai": {
                    "models": {
                        model: {
                            "id": model,
                            "endpoints": [ENDPOINT_ID],
                            "context_length": 8192,
                            "max_output_tokens": 1024,
                        }
                    }
                }
            },
        }
    )


def _response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="ok",
                    reasoning_content=None,
                    tool_calls=None,
                )
            )
        ],
        usage=None,
    )


def _bind_raw_client(client: Any, create: Callable[..., Any], model: str) -> None:
    client.endpoint = client.settings.get_endpoint(ENDPOINT_ID)
    client.model_id = model
    client.__dict__["raw_client"] = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gpt-5", True),
        ("gpt-5.6-luna", True),
        ("openai/gpt-5-mini", True),
        ("o1", True),
        ("o1-preview", True),
        ("openai/o3", True),
        ("o3-mini", True),
        ("openai/o3-mini-high", True),
        ("o3-pro", True),
        ("o4-mini", True),
        ("gpt-4o", False),
        ("o11", False),
        ("custom-o3", False),
    ],
)
def test_reasoning_model_detection_is_strict(model: str, expected: bool) -> None:
    assert _uses_max_completion_tokens(model) is expected


def test_sync_explicit_max_tokens_uses_completion_tokens_for_reasoning_model() -> None:
    model = "gpt-5"
    captured: dict[str, Any] = {}

    def create(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return _response()

    client = OpenAIChatClient(model=model, stream=False, settings=_settings(model))
    _bind_raw_client(client, create, model)
    client.create_completion(
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        skip_cutoff=True,
        max_tokens=123,
    )

    assert captured["max_tokens"] is OPENAI_NOT_GIVEN
    assert captured["max_completion_tokens"] == 123


def test_async_explicit_max_tokens_uses_completion_tokens_for_reasoning_model() -> None:
    model = "o1-preview"
    captured: dict[str, Any] = {}

    async def create(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return _response()

    async def run() -> None:
        client = AsyncOpenAIChatClient(model=model, stream=False, settings=_settings(model))
        _bind_raw_client(client, create, model)
        await client.create_completion(
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
            skip_cutoff=True,
            max_tokens=123,
        )

    asyncio.run(run())

    assert captured["max_tokens"] is OPENAI_NOT_GIVEN
    assert captured["max_completion_tokens"] == 123


@pytest.mark.parametrize("client_type, model", [("sync", "o3-mini"), ("async", "o4-mini")])
def test_explicit_completion_tokens_take_priority(client_type: str, model: str) -> None:
    captured: dict[str, Any] = {}

    if client_type == "sync":

        def create(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _response()

        client = OpenAIChatClient(model=model, stream=False, settings=_settings(model))
        _bind_raw_client(client, create, model)
        client.create_completion(
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
            skip_cutoff=True,
            max_tokens=123,
            max_completion_tokens=456,
        )
    else:

        async def create(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _response()

        async def run() -> None:
            client = AsyncOpenAIChatClient(model=model, stream=False, settings=_settings(model))
            _bind_raw_client(client, create, model)
            await client.create_completion(
                messages=[{"role": "user", "content": "hello"}],
                stream=False,
                skip_cutoff=True,
                max_tokens=123,
                max_completion_tokens=456,
            )

        asyncio.run(run())

    assert captured["max_tokens"] is OPENAI_NOT_GIVEN
    assert captured["max_completion_tokens"] == 456


@pytest.mark.parametrize("client_type", ["sync", "async"])
def test_legacy_model_keeps_max_tokens(client_type: str) -> None:
    model = "gpt-4o"
    captured: dict[str, Any] = {}

    if client_type == "sync":

        def create(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _response()

        client = OpenAIChatClient(model=model, stream=False, settings=_settings(model))
        _bind_raw_client(client, create, model)
        client.create_completion(
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
            skip_cutoff=True,
            max_tokens=123,
        )
    else:

        async def create(**kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return _response()

        async def run() -> None:
            client = AsyncOpenAIChatClient(model=model, stream=False, settings=_settings(model))
            _bind_raw_client(client, create, model)
            await client.create_completion(
                messages=[{"role": "user", "content": "hello"}],
                stream=False,
                skip_cutoff=True,
                max_tokens=123,
            )

        asyncio.run(run())

    assert captured["max_tokens"] == 123
    assert captured["max_completion_tokens"] is OPENAI_NOT_GIVEN


def test_sync_reasoning_model_auto_budget_uses_completion_tokens() -> None:
    model = "gpt-5"
    captured: dict[str, Any] = {}

    def create(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return _response()

    client = OpenAIChatClient(model=model, stream=False, settings=_settings(model))
    _bind_raw_client(client, create, model)
    client.create_completion(
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        skip_cutoff=True,
    )

    assert captured["max_tokens"] is OPENAI_NOT_GIVEN
    assert captured["max_completion_tokens"] == 1024


def test_async_reasoning_model_auto_budget_uses_completion_tokens() -> None:
    model = "o4-mini"
    captured: dict[str, Any] = {}

    async def create(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return _response()

    async def run() -> None:
        client = AsyncOpenAIChatClient(model=model, stream=False, settings=_settings(model))
        _bind_raw_client(client, create, model)
        await client.create_completion(
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
            skip_cutoff=True,
        )

    asyncio.run(run())

    assert captured["max_tokens"] is OPENAI_NOT_GIVEN
    assert captured["max_completion_tokens"] == 1024
