"""Independent sync and async decision clients."""

from __future__ import annotations

from typing import Any

import httpx2
from openai import AsyncOpenAI, OpenAI, DefaultHttpx2Client, DefaultAsyncHttpx2Client

from ..contract import load_catalog
from ..settings import Settings, normalize_settings, order_endpoints
from ..types.decision import DecisionRequest, DecisionResponse
from ..types.enums import BackendType
from ..types.settings import SettingsDict


def _resolve(backend: str, model: str, endpoint_id: str, settings: Settings):
    if backend != "openai":
        raise ValueError(f"Unsupported decision backend: {backend}")
    backend_settings = settings.get_decision_backend(backend)
    key = next((key for key, entry in backend_settings.models.items() if key == model or entry.id == model), model)
    configured = backend_settings.get_model_setting(key)
    if not configured.enabled:
        raise ValueError("Decision model is disabled")
    bindings = configured.endpoints
    if not bindings and settings.get_decision_backend(backend).default_endpoint:
        bindings = [settings.get_decision_backend(backend).default_endpoint]
    for binding in order_endpoints(bindings):
        binding_id = binding if isinstance(binding, str) else binding["endpoint_id"]
        if endpoint_id and binding_id != endpoint_id:
            continue
        if isinstance(binding, dict) and binding.get("enabled") is False:
            continue
        try:
            endpoint = settings.get_endpoint(binding_id)
        except ValueError:
            continue
        if not endpoint.enabled:
            continue
        if endpoint.endpoint_type not in (None, "default", "openai") or endpoint.is_azure or endpoint.is_vertex or endpoint.is_bedrock:
            raise ValueError("Decision client requires an OpenAI endpoint")
        capabilities = configured.capabilities.model_dump() if configured.capabilities else {}
        if isinstance(binding, dict):
            capabilities.update(binding.get("capabilities", {}))
        model_id = configured.id if isinstance(binding, str) else binding.get("model_id") or configured.id
        return endpoint, model_id, capabilities.get("decision_types")
    raise ValueError("Decision model has no enabled endpoint binding")


class _DecisionClientConfig:
    def __init__(
        self,
        backend: BackendType | str = "openai",
        model: str | None = None,
        *,
        settings: Settings | SettingsDict | None = None,
        endpoint_id: str = "",
        http_client=None,
        timeout: float = 60,
        max_retries: int = 0,
    ):
        self.backend = backend.value if isinstance(backend, BackendType) else str(backend).lower()
        self.settings = normalize_settings(settings)
        if model is None:
            models = load_catalog()["backends"].get(self.backend, {}).get("models", {})
            model = next((name for name, entry in models.items() if (entry.get("capabilities") or {}).get("decision_types")), None)
        if not model:
            raise ValueError("A decision model is required")
        self.model = model
        self.endpoint_id = endpoint_id
        self.http_client = http_client
        self.timeout = timeout
        self.max_retries = max_retries
        self._clients: dict[str, Any] = {}

    def _prepare(self, request: DecisionRequest):
        request = DecisionRequest.model_validate(request.model_dump()) if request.model else DecisionRequest.model_validate({**request.model_dump(), "model": self.model})
        endpoint, model_id, supported = _resolve(self.backend, request.model, self.endpoint_id, self.settings)
        if supported is None or any(question.type not in supported for question in request.questions):
            raise ValueError("Selected model does not declare support for the requested decision types")
        body = request.to_contract()
        body["model"] = model_id
        return request, endpoint, body

    def _options(self, endpoint):
        return {"api_key": endpoint.api_key, "base_url": endpoint.api_base, "timeout": self.timeout, "max_retries": self.max_retries, "default_headers": endpoint.headers}


class DecisionClient(_DecisionClientConfig):
    def create(self, request: DecisionRequest) -> DecisionResponse:
        request, endpoint, body = self._prepare(request)
        if endpoint.id not in self._clients:
            transport = self.http_client
            if transport is not None and not isinstance(transport, httpx2.Client):
                raise TypeError("http_client must be an httpx2.Client")
            if transport is None and endpoint.proxy:
                transport = DefaultHttpx2Client(proxy=endpoint.proxy)
            self._clients[endpoint.id] = OpenAI(**self._options(endpoint), http_client=transport)
        raw = self._clients[endpoint.id].post("/decisions", cast_to=dict[str, Any], body=body)
        response = DecisionResponse.model_validate(raw)
        response.validate_for(request)
        return response

    def close(self) -> None:
        if self.http_client is None:
            for client in self._clients.values():
                client.close()
        self._clients.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class AsyncDecisionClient(_DecisionClientConfig):
    async def create(self, request: DecisionRequest) -> DecisionResponse:
        request, endpoint, body = self._prepare(request)
        if endpoint.id not in self._clients:
            transport = self.http_client
            if transport is not None and not isinstance(transport, httpx2.AsyncClient):
                raise TypeError("http_client must be an httpx2.AsyncClient")
            if transport is None and endpoint.proxy:
                transport = DefaultAsyncHttpx2Client(proxy=endpoint.proxy)
            self._clients[endpoint.id] = AsyncOpenAI(**self._options(endpoint), http_client=transport)
        raw = await self._clients[endpoint.id].post("/decisions", cast_to=dict[str, Any], body=body)
        response = DecisionResponse.model_validate(raw)
        response.validate_for(request)
        return response

    async def close(self) -> None:
        if self.http_client is None:
            for client in self._clients.values():
                await client.close()
        self._clients.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


def create_decision_client(backend: BackendType | str = "openai", model: str | None = None, **kwargs) -> DecisionClient:
    return DecisionClient(backend, model, **kwargs)


def create_async_decision_client(backend: BackendType | str = "openai", model: str | None = None, **kwargs) -> AsyncDecisionClient:
    return AsyncDecisionClient(backend, model, **kwargs)


__all__ = ["DecisionClient", "AsyncDecisionClient", "create_decision_client", "create_async_decision_client"]
