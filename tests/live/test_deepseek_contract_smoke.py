"""Opt-in, provider-neutral DeepSeek non-streaming/streaming smoke check.

The script intentionally reports only response shape, usage counters, and
exception type. It never prints response content, settings, endpoint data, or
credentials.
"""

from __future__ import annotations

import json
import os
from typing import Any

from vv_llm import ChatRequest, ChatRequestOptions, ThinkingPreference
from vv_llm.chat_clients import BackendType, create_chat_client
from vv_llm.settings import settings

from live_common import load_live_settings

TRUTHY = {"1", "true", "yes", "on"}
MODEL = os.getenv("VV_LLM_MODEL", "deepseek-v4-flash").strip() or "deepseek-v4-flash"


def _usage(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    result: dict[str, int] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        number = getattr(value, field, None)
        if isinstance(number, int):
            result[field] = number
    return result or None


def _messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "Reply with exactly OK."}]


def _run_non_streaming() -> dict[str, Any]:
    try:
        client = create_chat_client(BackendType.DeepSeek, model=MODEL, stream=False)
        response = client.create(
            ChatRequest(
                model=MODEL,
                messages=_messages(),
                options=ChatRequestOptions(
                    thinking=ThinkingPreference.disabled(),
                    max_tokens=32,
                    timeout=60,
                ),
            )
        )
        return {
            "exit": 0,
            "response_type": type(response).__name__,
            "content_nonempty": bool(response.content),
            "usage": _usage(response.usage),
        }
    except Exception as exc:  # noqa: BLE001 - smoke output must remain structured and secret-free
        return {"exit": 1, "error_type": type(exc).__name__}


def _run_streaming() -> dict[str, Any]:
    try:
        client = create_chat_client(BackendType.DeepSeek, model=MODEL, stream=True)
        chunks = 0
        content_chars = 0
        reasoning_chars = 0
        usage = None
        stream = client.create(
            ChatRequest(
                model=MODEL,
                messages=_messages(),
                stream=True,
                options=ChatRequestOptions(
                    thinking=ThinkingPreference.disabled(),
                    stream_options={"include_usage": True},
                    max_tokens=32,
                    timeout=60,
                ),
            )
        )
        for chunk in stream:
            chunks += 1
            content_chars += len(chunk.content or "")
            reasoning_chars += len(chunk.reasoning_content or "")
            if chunk.usage is not None:
                usage = _usage(chunk.usage)
        return {
            "exit": 0,
            "chunks": chunks,
            "content_chars": content_chars,
            "reasoning_chars": reasoning_chars,
            "usage": usage,
        }
    except Exception as exc:  # noqa: BLE001 - smoke output must remain structured and secret-free
        return {"exit": 1, "error_type": type(exc).__name__}


def main() -> int:
    if os.getenv("VV_LLM_RUN_LIVE_TESTS", "").strip().lower() not in TRUTHY:
        print("Live smoke disabled. Set VV_LLM_RUN_LIVE_TESTS=1 or use run_live_tests.py.")
        return 1

    try:
        load_live_settings(settings)
    except Exception as exc:  # noqa: BLE001 - keep configuration failures secret-free
        print(json.dumps({"provider": "deepseek", "model": MODEL, "load_exit": 1, "error_type": type(exc).__name__}, sort_keys=True))
        return 1

    non_stream = _run_non_streaming()
    stream = _run_streaming()
    result = {
        "provider": "deepseek",
        "model": MODEL,
        "non_stream": non_stream,
        "stream": stream,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if non_stream["exit"] == 0 and stream["exit"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
