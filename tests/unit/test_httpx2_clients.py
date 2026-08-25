from __future__ import annotations

import asyncio

import httpx2
import pytest

from vv_llm.chat_clients import BackendType, create_async_chat_client, create_chat_client


def _settings(backend: str, endpoint_type: str) -> dict:
    model = "gpt-test" if backend == "openai" else "claude-test"
    return {
        "rate_limit": {"enabled": False},
        "endpoints": [
            {
                "id": f"{backend}-endpoint",
                "api_base": "https://example.test/v1",
                "api_key": "test-key",
                "endpoint_type": endpoint_type,
            }
        ],
        "backends": {
            backend: {
                "models": {
                    model: {
                        "id": model,
                        "endpoints": [f"{backend}-endpoint"],
                        "context_length": 8192,
                        "max_output_tokens": 1024,
                    }
                }
            }
        },
    }


def _transport() -> httpx2.MockTransport:
    return httpx2.MockTransport(
        lambda request: httpx2.Response(
            200,
            json={"id": "offline", "choices": [], "model": "test"},
            request=request,
        )
    )


def test_openai_sync_accepts_httpx2_custom_client_without_network() -> None:
    custom = httpx2.Client(transport=_transport())
    client = create_chat_client(
        BackendType.OpenAI,
        model="gpt-test",
        random_endpoint=False,
        endpoint_id="openai-endpoint",
        http_client=custom,
        settings=_settings("openai", "default"),
    )

    raw = client.raw_client

    assert client.http_client is custom
    assert raw._client is custom
    custom.close()


def test_openai_async_accepts_httpx2_custom_client_without_network() -> None:
    async def run() -> None:
        custom = httpx2.AsyncClient(transport=_transport())
        client = create_async_chat_client(
            BackendType.OpenAI,
            model="gpt-test",
            random_endpoint=False,
            endpoint_id="openai-endpoint",
            http_client=custom,
            settings=_settings("openai", "default"),
        )

        raw = client.raw_client

        assert client.http_client is custom
        assert raw._client is custom
        await custom.aclose()

    asyncio.run(run())


def test_anthropic_sync_accepts_httpx2_custom_client_without_network() -> None:
    custom = httpx2.Client(transport=_transport())
    client = create_chat_client(
        BackendType.Anthropic,
        model="claude-test",
        random_endpoint=False,
        endpoint_id="anthropic-endpoint",
        http_client=custom,
        settings=_settings("anthropic", "anthropic"),
    )

    raw = client.raw_client

    assert client.http_client is custom
    assert raw._client is custom
    custom.close()


def test_anthropic_async_accepts_httpx2_custom_client_without_network() -> None:
    async def run() -> None:
        custom = httpx2.AsyncClient(transport=_transport())
        client = create_async_chat_client(
            BackendType.Anthropic,
            model="claude-test",
            random_endpoint=False,
            endpoint_id="anthropic-endpoint",
            http_client=custom,
            settings=_settings("anthropic", "anthropic"),
        )

        raw = client.raw_client

        assert client.http_client is custom
        assert raw._client is custom
        await custom.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("backend", "model", "settings_backend", "endpoint_type"),
    [
        (BackendType.OpenAI, "gpt-test", "openai", "default"),
        (BackendType.Anthropic, "claude-test", "anthropic", "anthropic"),
    ],
)
def test_sync_default_sdk_client_is_constructed_without_custom_transport(
    backend: BackendType,
    model: str,
    settings_backend: str,
    endpoint_type: str,
) -> None:
    client = create_chat_client(
        backend,
        model=model,
        random_endpoint=False,
        endpoint_id=f"{settings_backend}-endpoint",
        settings=_settings(settings_backend, endpoint_type),
    )

    raw = client.raw_client

    assert client.http_client is None
    assert raw._client is not None


@pytest.mark.parametrize(
    ("backend", "model", "settings_backend", "endpoint_type"),
    [
        (BackendType.OpenAI, "gpt-test", "openai", "default"),
        (BackendType.Anthropic, "claude-test", "anthropic", "anthropic"),
    ],
)
def test_async_default_sdk_client_is_constructed_without_custom_transport(
    backend: BackendType,
    model: str,
    settings_backend: str,
    endpoint_type: str,
) -> None:
    async def run() -> None:
        client = create_async_chat_client(
            backend,
            model=model,
            random_endpoint=False,
            endpoint_id=f"{settings_backend}-endpoint",
            settings=_settings(settings_backend, endpoint_type),
        )

        raw = client.raw_client

        assert client.http_client is None
        assert raw._client is not None
        await raw.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("backend", "model", "module_name"),
    [
        (BackendType.OpenAI, "gpt-test", "vv_llm.chat_clients.openai_compatible_client"),
        (BackendType.Anthropic, "claude-test", "vv_llm.chat_clients.anthropic_client"),
    ],
)
def test_proxy_endpoint_builds_httpx2_client(
    backend: BackendType,
    model: str,
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    module = importlib.import_module(module_name)
    captured: dict[str, object] = {}
    default_name = "DefaultHttpx2Client" if backend is BackendType.OpenAI else "DefaultHttpxClient"

    def recording_factory(*args, **kwargs):
        captured.update(kwargs)
        return httpx2.Client(*args, **kwargs)

    monkeypatch.setattr(module, default_name, recording_factory)
    settings = _settings("openai" if backend is BackendType.OpenAI else "anthropic", "default")
    settings["endpoints"][0]["proxy"] = "http://proxy.example.test:8080"

    client = create_chat_client(
        backend,
        model=model,
        random_endpoint=False,
        endpoint_id=f"{'openai' if backend is BackendType.OpenAI else 'anthropic'}-endpoint",
        settings=settings,
    )
    raw = client.raw_client

    assert raw._client is client.http_client
    assert captured["proxy"] == "http://proxy.example.test:8080"
    client.http_client.close()


@pytest.mark.parametrize(
    ("backend", "model", "module_name"),
    [
        (BackendType.OpenAI, "gpt-test", "vv_llm.chat_clients.openai_compatible_client"),
        (BackendType.Anthropic, "claude-test", "vv_llm.chat_clients.anthropic_client"),
    ],
)
def test_async_proxy_endpoint_builds_httpx2_client(
    backend: BackendType,
    model: str,
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    module = importlib.import_module(module_name)
    captured: dict[str, object] = {}
    default_name = "DefaultAsyncHttpx2Client" if backend is BackendType.OpenAI else "DefaultAsyncHttpxClient"

    def recording_factory(*args, **kwargs):
        captured.update(kwargs)
        return httpx2.AsyncClient(*args, **kwargs)

    monkeypatch.setattr(module, default_name, recording_factory)
    settings_backend = "openai" if backend is BackendType.OpenAI else "anthropic"
    endpoint_type = "default" if backend is BackendType.OpenAI else "anthropic"
    settings = _settings(settings_backend, endpoint_type)
    settings["endpoints"][0]["proxy"] = "http://proxy.example.test:8080"

    async def run() -> None:
        client = create_async_chat_client(
            backend,
            model=model,
            random_endpoint=False,
            endpoint_id=f"{settings_backend}-endpoint",
            settings=settings,
        )
        raw = client.raw_client

        assert raw._client is client.http_client
        assert captured["proxy"] == "http://proxy.example.test:8080"
        await client.http_client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("backend", "model", "settings_backend"),
    [
        (BackendType.OpenAI, "gpt-test", "openai"),
        (BackendType.Anthropic, "claude-test", "anthropic"),
    ],
)
def test_incompatible_http_client_is_rejected_at_vv_llm_boundary(
    backend: BackendType,
    model: str,
    settings_backend: str,
) -> None:
    incompatible_client = object()
    with pytest.raises(TypeError, match="httpx2|http_client"):
        client = create_chat_client(
            backend,
            model=model,
            random_endpoint=False,
            endpoint_id=f"{settings_backend}-endpoint",
            http_client=incompatible_client,
            settings=_settings(settings_backend, "anthropic" if backend is BackendType.Anthropic else "default"),
        )
        _ = client.raw_client
