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


def load_deepseek_client(*, asynchronous: bool = False) -> Any:
    settings_path = os.environ.get("VV_LLM_SETTINGS_JSON")
    if not settings_path:
        raise RuntimeError("set VV_LLM_SETTINGS_JSON to a vv-llm settings JSON file")

    payload = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    settings.load(payload)
    model = os.environ.get("VV_LLM_MODEL", "deepseek-v4-flash")
    factory = create_async_chat_client if asynchronous else create_chat_client
    return factory(BackendType.DeepSeek, model=model)
