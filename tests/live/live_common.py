from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from typing import TypeVar

from vv_llm.types.enums import BackendType

TRUTHY = {"1", "true", "yes", "on"}
T = TypeVar("T")


def _is_truthy(value: str | None) -> bool:
    return bool(value and value.strip().lower() in TRUTHY)


def _has_live_credentials(sample_settings: dict) -> bool:
    for endpoint in sample_settings.get("endpoints", []):
        if not isinstance(endpoint, dict):
            continue

        api_key = endpoint.get("api_key")
        if isinstance(api_key, str) and api_key.strip() and not api_key.strip().startswith("YOUR_"):
            return True

        credentials = endpoint.get("credentials")
        if isinstance(credentials, dict) and any(bool(value) for value in credentials.values()):
            return True

    return False


def load_live_settings(settings_obj, require_credentials: bool = True) -> None:
    from sample_settings import sample_settings

    settings_obj.load(sample_settings)
    allow_empty = _is_truthy(os.getenv("VV_LLM_ALLOW_EMPTY_KEYS"))
    if require_credentials and not allow_empty and not _has_live_credentials(sample_settings):
        raise RuntimeError("Live settings appear to have no usable API credentials. Create tests/dev_settings.py or set VV_LLM_ALLOW_EMPTY_KEYS=1 to bypass this check.")


def resolve_backend_model(
    default_backend: BackendType,
    default_model: str,
    *,
    presets: Mapping[str, tuple[BackendType, str]] | None = None,
) -> tuple[BackendType, str]:
    backend = default_backend
    model = default_model

    preset_name = os.getenv("VV_LLM_MODEL_PRESET", "").strip()
    if preset_name:
        if not presets or preset_name not in presets:
            available = ", ".join(sorted(presets)) if presets else "(none)"
            raise ValueError(f"Unknown VV_LLM_MODEL_PRESET={preset_name!r}. Available presets: {available}")
        backend, model = presets[preset_name]

    backend_raw = os.getenv("VV_LLM_BACKEND", "").strip().lower()
    if backend_raw:
        backend = BackendType(backend_raw)

    model_raw = os.getenv("VV_LLM_MODEL", "").strip()
    if model_raw:
        model = model_raw

    print(f"[live] backend={backend.value} model={model}")
    return backend, model


def resolve_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return _is_truthy(raw)


def run_with_timer(label: str, fn: Callable[[], T]) -> T:
    start = time.perf_counter()
    try:
        return fn()
    finally:
        elapsed = time.perf_counter() - start
        print(f"[live] {label} finished in {elapsed:.2f}s")
