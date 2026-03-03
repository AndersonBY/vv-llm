from __future__ import annotations

import asyncio
import random
import re
import time
from collections import defaultdict
from typing import Any, cast

import httpx

from ..settings import Settings, normalize_settings
from ..types.llm_parameters import EndpointSetting, RetrievalBackendSettings
from ..types.settings import EndpointOptionDict, SettingsDict
from ..utilities.rate_limiter import AsyncDiskCacheRateLimiter, AsyncMemoryRateLimiter, AsyncRedisRateLimiter
from ..utilities.rate_limiter import SyncDiskCacheRateLimiter, SyncMemoryRateLimiter, SyncRedisRateLimiter

_PLACEHOLDER_RE = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class BaseRetrievalClient:
    def __init__(
        self,
        *,
        model: str,
        backend_name: str,
        backend_settings: RetrievalBackendSettings,
        random_endpoint: bool = True,
        endpoint_id: str = "",
        http_client: httpx.Client | None = None,
        settings: Settings | SettingsDict | None = None,
    ):
        self.model = model or self._resolve_default_model(backend_settings)
        self.backend_name = backend_name
        self.backend_settings = backend_settings
        self.random_endpoint = random_endpoint
        self.endpoint_id = endpoint_id
        self.http_client = http_client

        self.settings: Settings = normalize_settings(settings)

        self.rate_limiter = self._init_rate_limiter()
        self.active_requests = defaultdict(int)
        self.rpm = None
        self.tpm = None
        self.concurrent_requests = None

        self.model_setting = self.backend_settings.get_model_setting(self.model)
        self.model_id = self.model_setting.id
        self.endpoint: EndpointSetting | None = None

        if endpoint_id:
            self.random_endpoint = False
            self.endpoint = self.settings.get_endpoint(self.endpoint_id)
            self._set_model_id_by_endpoint_id(self.endpoint_id)

    @staticmethod
    def _resolve_default_model(backend_settings: RetrievalBackendSettings) -> str:
        for model_name, model_setting in backend_settings.models.items():
            if model_setting.enabled:
                return model_name
        if backend_settings.models:
            return next(iter(backend_settings.models.keys()))
        raise ValueError("No models configured for retrieval backend.")

    def _init_rate_limiter(self):
        if not self.settings.rate_limit:
            return None
        if not self.settings.rate_limit.enabled:
            return None

        if self.settings.rate_limit.backend == "memory":
            return SyncMemoryRateLimiter()
        if self.settings.rate_limit.backend == "redis":
            if not self.settings.rate_limit.redis:
                raise ValueError("Redis settings must be provided if Redis backend is selected.")
            return SyncRedisRateLimiter(
                host=self.settings.rate_limit.redis.host,
                port=self.settings.rate_limit.redis.port,
                db=self.settings.rate_limit.redis.db,
            )
        if self.settings.rate_limit.backend == "diskcache":
            if not self.settings.rate_limit.diskcache:
                raise ValueError("Diskcache settings must be provided if Diskcache backend is selected.")
            return SyncDiskCacheRateLimiter(
                cache_dir=self.settings.rate_limit.diskcache.cache_dir,
            )
        return None

    @staticmethod
    def _estimate_tokens(payload: Any) -> int:
        text = str(payload)
        cjk_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f" or "\uff00" <= ch <= "\uffef")
        ascii_chars = len(text) - cjk_chars
        return max(int(cjk_chars * 1.5 + ascii_chars * 0.25), 1)

    def _acquire_rate_limit(self, endpoint: EndpointSetting | None, payload: Any):
        if endpoint is None:
            return

        key = f"{endpoint.id}:{self.model}"
        rpm = self.rpm or endpoint.rpm or (self.settings.rate_limit.default_rpm if self.settings.rate_limit else 60)
        tpm = self.tpm or endpoint.tpm or (self.settings.rate_limit.default_tpm if self.settings.rate_limit else 1000000)
        est_tokens = self._estimate_tokens(payload)

        while self.rate_limiter:
            allowed, wait_time = self.rate_limiter.check_limit(key, rpm, tpm, est_tokens)
            if allowed:
                break
            time.sleep(wait_time)

    def _get_available_endpoints(self, model_endpoints: list[str | EndpointOptionDict]) -> list[str | EndpointOptionDict]:
        available_endpoints = []
        for endpoint_option in model_endpoints:
            if isinstance(endpoint_option, str):
                endpoint_id = endpoint_option
            else:
                endpoint_id = endpoint_option.get("endpoint_id")
                if not isinstance(endpoint_id, str):
                    continue
            try:
                endpoint = self.settings.get_endpoint(endpoint_id)
            except ValueError:
                continue
            if endpoint.enabled:
                available_endpoints.append(endpoint_option)
        return available_endpoints

    def _set_model_id_by_endpoint_id(self, endpoint_id: str) -> str:
        self.model_id = self.model_setting.id
        for endpoint_option in self.model_setting.endpoints:
            if isinstance(endpoint_option, dict) and endpoint_option.get("endpoint_id") == endpoint_id:
                self.model_id = endpoint_option.get("model_id", self.model_setting.id)
                break
        return self.model_id

    def _set_endpoint(self) -> tuple[EndpointSetting, str]:
        if self.endpoint is None:
            if self.random_endpoint:
                available_endpoints = self._get_available_endpoints(
                    cast(list[str | EndpointOptionDict], self.model_setting.endpoints)
                )
                if not available_endpoints:
                    raise ValueError(f"No enabled endpoints available for model {self.model}")

                endpoint_option = random.choice(available_endpoints)
                if isinstance(endpoint_option, dict):
                    self.endpoint_id = endpoint_option["endpoint_id"]
                    self.model_id = endpoint_option.get("model_id", self.model_setting.id)
                    self.rpm = endpoint_option.get("rpm", None)
                    self.tpm = endpoint_option.get("tpm", None)
                    self.concurrent_requests = endpoint_option.get("concurrent_requests", None)
                else:
                    self.endpoint_id = endpoint_option
                    self.model_id = self.model_setting.id
                self.endpoint = self.settings.get_endpoint(self.endpoint_id)
            else:
                if not self.endpoint_id:
                    raise ValueError("endpoint_id is required when random_endpoint is False")
                self.endpoint = self.settings.get_endpoint(self.endpoint_id)
                if not self.endpoint.enabled:
                    raise ValueError(f"Endpoint {self.endpoint_id} is disabled")
                self._set_model_id_by_endpoint_id(self.endpoint_id)
        else:
            if not self.endpoint.enabled:
                raise ValueError(f"Endpoint {self.endpoint.id} is disabled")
            self.endpoint_id = self.endpoint.id
            self._set_model_id_by_endpoint_id(self.endpoint_id)

        return self.endpoint, self.model_id


class BaseAsyncRetrievalClient:
    def __init__(
        self,
        *,
        model: str,
        backend_name: str,
        backend_settings: RetrievalBackendSettings,
        random_endpoint: bool = True,
        endpoint_id: str = "",
        http_client: httpx.AsyncClient | None = None,
        settings: Settings | SettingsDict | None = None,
    ):
        self.model = model or self._resolve_default_model(backend_settings)
        self.backend_name = backend_name
        self.backend_settings = backend_settings
        self.random_endpoint = random_endpoint
        self.endpoint_id = endpoint_id
        self.http_client = http_client

        self.settings: Settings = normalize_settings(settings)

        self.rate_limiter = self._init_rate_limiter()
        self.active_requests = defaultdict(int)
        self.rpm = None
        self.tpm = None
        self.concurrent_requests = None

        self.model_setting = self.backend_settings.get_model_setting(self.model)
        self.model_id = self.model_setting.id
        self.endpoint: EndpointSetting | None = None

        if endpoint_id:
            self.random_endpoint = False
            self.endpoint = self.settings.get_endpoint(self.endpoint_id)
            self._set_model_id_by_endpoint_id(self.endpoint_id)

    @staticmethod
    def _resolve_default_model(backend_settings: RetrievalBackendSettings) -> str:
        for model_name, model_setting in backend_settings.models.items():
            if model_setting.enabled:
                return model_name
        if backend_settings.models:
            return next(iter(backend_settings.models.keys()))
        raise ValueError("No models configured for retrieval backend.")

    def _init_rate_limiter(self):
        if not self.settings.rate_limit:
            return None
        if not self.settings.rate_limit.enabled:
            return None

        if self.settings.rate_limit.backend == "memory":
            return AsyncMemoryRateLimiter()
        if self.settings.rate_limit.backend == "redis":
            if not self.settings.rate_limit.redis:
                raise ValueError("Redis settings must be provided if Redis backend is selected.")
            return AsyncRedisRateLimiter(
                host=self.settings.rate_limit.redis.host,
                port=self.settings.rate_limit.redis.port,
                db=self.settings.rate_limit.redis.db,
            )
        if self.settings.rate_limit.backend == "diskcache":
            if not self.settings.rate_limit.diskcache:
                raise ValueError("Diskcache settings must be provided if Diskcache backend is selected.")
            return AsyncDiskCacheRateLimiter(
                cache_dir=self.settings.rate_limit.diskcache.cache_dir,
            )
        return None

    @staticmethod
    def _estimate_tokens(payload: Any) -> int:
        text = str(payload)
        cjk_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f" or "\uff00" <= ch <= "\uffef")
        ascii_chars = len(text) - cjk_chars
        return max(int(cjk_chars * 1.5 + ascii_chars * 0.25), 1)

    async def _acquire_rate_limit(self, endpoint: EndpointSetting | None, payload: Any):
        if endpoint is None:
            return

        key = f"{endpoint.id}:{self.model}"
        rpm = self.rpm or endpoint.rpm or (self.settings.rate_limit.default_rpm if self.settings.rate_limit else 60)
        tpm = self.tpm or endpoint.tpm or (self.settings.rate_limit.default_tpm if self.settings.rate_limit else 1000000)
        est_tokens = self._estimate_tokens(payload)

        while self.rate_limiter:
            allowed, wait_time = await self.rate_limiter.check_limit(key, rpm, tpm, est_tokens)
            if allowed:
                break
            await asyncio.sleep(wait_time)

    def _get_available_endpoints(self, model_endpoints: list[str | EndpointOptionDict]) -> list[str | EndpointOptionDict]:
        available_endpoints = []
        for endpoint_option in model_endpoints:
            if isinstance(endpoint_option, str):
                endpoint_id = endpoint_option
            else:
                endpoint_id = endpoint_option.get("endpoint_id")
                if not isinstance(endpoint_id, str):
                    continue
            try:
                endpoint = self.settings.get_endpoint(endpoint_id)
            except ValueError:
                continue
            if endpoint.enabled:
                available_endpoints.append(endpoint_option)
        return available_endpoints

    def _set_model_id_by_endpoint_id(self, endpoint_id: str) -> str:
        self.model_id = self.model_setting.id
        for endpoint_option in self.model_setting.endpoints:
            if isinstance(endpoint_option, dict) and endpoint_option.get("endpoint_id") == endpoint_id:
                self.model_id = endpoint_option.get("model_id", self.model_setting.id)
                break
        return self.model_id

    def _set_endpoint(self) -> tuple[EndpointSetting, str]:
        if self.endpoint is None:
            if self.random_endpoint:
                available_endpoints = self._get_available_endpoints(
                    cast(list[str | EndpointOptionDict], self.model_setting.endpoints)
                )
                if not available_endpoints:
                    raise ValueError(f"No enabled endpoints available for model {self.model}")

                endpoint_option = random.choice(available_endpoints)
                if isinstance(endpoint_option, dict):
                    self.endpoint_id = endpoint_option["endpoint_id"]
                    self.model_id = endpoint_option.get("model_id", self.model_setting.id)
                    self.rpm = endpoint_option.get("rpm", None)
                    self.tpm = endpoint_option.get("tpm", None)
                    self.concurrent_requests = endpoint_option.get("concurrent_requests", None)
                else:
                    self.endpoint_id = endpoint_option
                    self.model_id = self.model_setting.id
                self.endpoint = self.settings.get_endpoint(self.endpoint_id)
            else:
                if not self.endpoint_id:
                    raise ValueError("endpoint_id is required when random_endpoint is False")
                self.endpoint = self.settings.get_endpoint(self.endpoint_id)
                if not self.endpoint.enabled:
                    raise ValueError(f"Endpoint {self.endpoint_id} is disabled")
                self._set_model_id_by_endpoint_id(self.endpoint_id)
        else:
            if not self.endpoint.enabled:
                raise ValueError(f"Endpoint {self.endpoint.id} is disabled")
            self.endpoint_id = self.endpoint.id
            self._set_model_id_by_endpoint_id(self.endpoint_id)

        return self.endpoint, self.model_id


def build_url(api_base: str | None, path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if api_base is None:
        raise ValueError("api_base is required for relative request paths.")
    return f"{api_base.rstrip('/')}/{path.lstrip('/')}"


def render_template(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {k: render_template(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [render_template(v, context) for v in value]
    if isinstance(value, str):
        matches = list(_PLACEHOLDER_RE.finditer(value))
        if not matches:
            return value
        if len(matches) == 1 and matches[0].span() == (0, len(value)):
            return context.get(matches[0].group(1))

        result = value
        for match in matches:
            key = match.group(1)
            result = result.replace(f"${{{key}}}", str(context.get(key, "")))
        return result
    return value


def _tokenize_path(path: str) -> list[tuple[str, str | int]]:
    if path == "$":
        return []
    if not path.startswith("$"):
        raise ValueError(f"Path must start with '$': {path}")

    tokens: list[tuple[str, str | int]] = []
    i = 1
    n = len(path)

    while i < n:
        ch = path[i]
        if ch == ".":
            i += 1
            start = i
            while i < n and path[i] not in ".[":
                i += 1
            key = path[start:i]
            if key:
                tokens.append(("key", key))
            continue

        if ch == "[":
            j = path.find("]", i)
            if j == -1:
                raise ValueError(f"Invalid path: {path}")
            content = path[i + 1 : j]
            if content == "*":
                tokens.append(("all", "*"))
            elif content.isdigit():
                tokens.append(("index", int(content)))
            elif (content.startswith("'") and content.endswith("'")) or (content.startswith('"') and content.endswith('"')):
                tokens.append(("key", content[1:-1]))
            else:
                raise ValueError(f"Unsupported bracket selector in path: {path}")
            i = j + 1
            continue

        i += 1

    return tokens


def extract_json_path(data: Any, path: str | None) -> Any:
    if path is None:
        return None

    tokens = _tokenize_path(path)
    values = [data]
    has_all = any(kind == "all" for kind, _ in tokens)

    for kind, token in tokens:
        next_values: list[Any] = []
        for current in values:
            if kind == "key":
                if isinstance(current, dict) and token in current:
                    next_values.append(current[token])
            elif kind == "index":
                if isinstance(current, list) and isinstance(token, int) and 0 <= token < len(current):
                    next_values.append(current[token])
            elif kind == "all":
                if isinstance(current, list):
                    next_values.extend(current)
        values = next_values

    if not values:
        return None
    if has_all:
        return values
    if len(values) == 1:
        return values[0]
    return values
