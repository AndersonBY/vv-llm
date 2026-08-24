from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from vv_llm.chat_clients import (
    BackendType,
    create_async_chat_client,
    create_chat_client,
)
from vv_llm.settings import settings


def load_client(*, asynchronous: bool = False) -> Any:
    settings_path = os.environ.get("VV_LLM_SETTINGS_JSON")
    if not settings_path:
        raise RuntimeError("set VV_LLM_SETTINGS_JSON to a vv-llm settings JSON file")

    try:
        payload = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("could not load VV_LLM_SETTINGS_JSON") from exc
    settings.load(payload)
    backend_name = os.environ.get("VV_LLM_BACKEND", "openai").strip().lower()
    try:
        backend = BackendType(backend_name)
    except ValueError as exc:
        raise RuntimeError("VV_LLM_BACKEND must name a supported backend") from exc

    model = os.environ.get("VV_LLM_MODEL") or None
    factory = create_async_chat_client if asynchronous else create_chat_client
    try:
        return factory(backend, model=model)
    except Exception as exc:
        raise RuntimeError("could not create the configured chat client") from exc
