"""Opt-in Responses API tool-call smoke check.

Runs a complete two-turn tool call through both the non-streaming and streaming
paths and reports only shape flags. It never prints response content, settings,
endpoint data, or credentials.

The selected endpoint is forced to the Responses transport for this run; set
``VV_LLM_ENDPOINT`` to pin a specific endpoint, otherwise the model's first
enabled binding is used.
"""

from __future__ import annotations

import json
import os
from typing import Any

from vv_llm.chat_clients import BackendType, create_chat_client
from vv_llm.settings import settings

from live_common import load_live_settings

TRUTHY = {"1", "true", "yes", "on"}
BACKEND = os.getenv("VV_LLM_BACKEND", "openai").strip() or "openai"
MODEL = os.getenv("VV_LLM_MODEL", "gpt-6-astra").strip() or "gpt-6-astra"
ENDPOINT = os.getenv("VV_LLM_ENDPOINT", "").strip()
QUESTION = "What is the weather in San Francisco? Call the get_weather tool."
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]


def _endpoint_id(backend: BackendType) -> str:
    if ENDPOINT:
        return ENDPOINT
    model_setting = settings.get_backend(backend).models[MODEL]
    binding = model_setting.endpoints[0]
    return binding if isinstance(binding, str) else binding["endpoint_id"]


def _client(stream: bool) -> Any:
    backend = BackendType(BACKEND)
    endpoint_id = _endpoint_id(backend)
    # The smoke check exercises the Responses adapter regardless of the stored transport flag.
    settings.get_endpoint(endpoint_id).response_api = True
    client = create_chat_client(backend=backend, model=MODEL, endpoint_id=endpoint_id, random_endpoint=False, settings=settings, stream=stream)
    client.raw_client.max_retries = 0
    client.raw_client.timeout = 180
    return client


def _tool_result(call_id: str, arguments: str) -> dict[str, str]:
    try:
        arguments_data = json.loads(arguments)
    except json.JSONDecodeError:
        arguments_data = {}
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps({"city": arguments_data.get("city", "unknown"), "temp_c": 18})}


def _history(turn_one: Any, call_id: str, arguments: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": QUESTION},
        {"role": "assistant", "content": turn_one.content, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "get_weather", "arguments": arguments}}]},
        _tool_result(call_id, arguments),
    ]


def _run_non_streaming() -> dict[str, Any]:
    client = None
    try:
        client = _client(stream=False)
        turn_one = client.create_completion(messages=[{"role": "user", "content": QUESTION}], tools=TOOLS, stream=False, max_completion_tokens=1024, skip_cutoff=True, timeout=180)
        calls = turn_one.tool_calls or []
        if not calls:
            return {"exit": 1, "error_type": "MissingToolCall"}
        call = calls[0]
        call_id_prefix_ok = isinstance(call.id, str) and not call.id.startswith("fc_")
        try:
            json.loads(call.function.arguments)
            arguments_json_ok = True
        except json.JSONDecodeError:
            arguments_json_ok = False
        turn_two = client.create_completion(
            messages=_history(turn_one, call.id, call.function.arguments), tools=TOOLS, stream=False, max_completion_tokens=1024, skip_cutoff=True, timeout=180
        )
        return {
            "exit": 0 if call_id_prefix_ok and arguments_json_ok and bool(turn_two.content) else 1,
            "tool_calls": len(calls),
            "call_id_prefix_ok": call_id_prefix_ok,
            "arguments_json_ok": arguments_json_ok,
            "second_turn_content": bool(turn_two.content),
        }
    except Exception as exc:  # noqa: BLE001 - smoke output must remain structured and secret-free
        return {"exit": 1, "error_type": type(exc).__name__, "status": getattr(exc, "status_code", None)}
    finally:
        if client is not None:
            client.raw_client.close()


def _run_streaming() -> dict[str, Any]:
    client = None
    try:
        client = _client(stream=True)
        tool_calls: dict[int, dict[str, Any]] = {}
        content_chars = 0
        for delta in client.create_completion(
            messages=[{"role": "user", "content": QUESTION}], tools=TOOLS, stream=True, max_completion_tokens=1024, skip_cutoff=True, timeout=180
        ):
            content_chars += len(delta.content or "")
            for call in delta.tool_calls or []:
                entry = tool_calls.setdefault(call.index, {"id": None, "name": None, "arguments": ""})
                if call.id:
                    entry["id"] = call.id
                if call.function and call.function.name:
                    entry["name"] = call.function.name
                if call.function and call.function.arguments:
                    entry["arguments"] += call.function.arguments
        if not tool_calls:
            return {"exit": 1, "error_type": "MissingToolCall"}
        first = tool_calls[0]
        call_id_prefix_ok = isinstance(first["id"], str) and not first["id"].startswith("fc_")
        try:
            json.loads(first["arguments"])
            arguments_json_ok = True
        except json.JSONDecodeError:
            arguments_json_ok = False
        history = [
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": "", "tool_calls": [{"id": first["id"], "type": "function", "function": {"name": first["name"], "arguments": first["arguments"]}}]},
            _tool_result(first["id"], first["arguments"]),
        ]
        second_turn_chars = 0
        for delta in client.create_completion(messages=history, tools=TOOLS, stream=True, max_completion_tokens=1024, skip_cutoff=True, timeout=180):
            second_turn_chars += len(delta.content or "")
        return {
            "exit": 0 if call_id_prefix_ok and arguments_json_ok and second_turn_chars > 0 else 1,
            "tool_calls": len(tool_calls),
            "content_chars": content_chars,
            "call_id_prefix_ok": call_id_prefix_ok,
            "arguments_json_ok": arguments_json_ok,
            "second_turn_chars": second_turn_chars,
        }
    except Exception as exc:  # noqa: BLE001 - smoke output must remain structured and secret-free
        return {"exit": 1, "error_type": type(exc).__name__, "status": getattr(exc, "status_code", None)}
    finally:
        if client is not None:
            client.raw_client.close()


def main() -> int:
    if os.getenv("VV_LLM_RUN_LIVE_TESTS", "").strip().lower() not in TRUTHY:
        print("Live smoke disabled. Set VV_LLM_RUN_LIVE_TESTS=1 or use run_live_tests.py.")
        return 1

    try:
        load_live_settings(settings)
    except Exception as exc:  # noqa: BLE001 - keep configuration failures secret-free
        print(json.dumps({"provider": BACKEND, "model": MODEL, "load_exit": 1, "error_type": type(exc).__name__}, sort_keys=True))
        return 1

    non_stream = _run_non_streaming()
    stream = _run_streaming()
    print(json.dumps({"provider": BACKEND, "model": MODEL, "non_stream": non_stream, "stream": stream}, sort_keys=True))
    return 0 if non_stream["exit"] == 0 and stream["exit"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
