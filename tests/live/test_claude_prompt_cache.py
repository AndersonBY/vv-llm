# @Author: Bi Ying
# @Date:   2024-07-27 11:51:28
from __future__ import annotations

import time
from typing import Any

from vv_llm.chat_clients import BackendType, create_chat_client, format_messages
from vv_llm.settings import settings
from vv_llm.types.llm_parameters import NOT_GIVEN, Usage

try:
    from .live_common import load_live_settings, resolve_backend_model
except ImportError:
    from live_common import load_live_settings, resolve_backend_model


load_live_settings(settings)

DEFAULT_MODEL = "claude-sonnet-4-6"
PRESETS = {
    "sonnet-4.5": (BackendType.Anthropic, "claude-sonnet-4-5-20250929"),
    "sonnet-4.6": (BackendType.Anthropic, "claude-sonnet-4-6"),
    "haiku-3.5": (BackendType.Anthropic, "claude-3-5-haiku-20241022"),
}

CACHE_REGRESSION_DOCUMENT = "\n".join(
    f"Section {index:03d}: Prompt caching regression text. "
    "This prefix is intentionally repetitive so repeated requests can reuse the same cached prompt."
    for index in range(1, 181)
)

# 测试场景1: System message 中的 cache_control (列表格式)
messages_system_cache = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "As an expert in SQLite, you are expected to utilize your knowledge to craft SQL queries that are in strict adherence with SQLite syntax standards when responding to inquiries.\n\nThe table structure is as follows:\n```sql\nCREATE TABLE 历年能源消费构成 (\n    年份        INTEGER,\n    煤炭        INTEGER,\n    石油        INTEGER,\n    天然气       INTEGER,\n    一次电力及其他能源 INTEGER\n);\n```",
                "cache_control": {"type": "ephemeral"},
            }
        ],
    },
    {
        "role": "user",
        "content": "筛出2005年到2010年之间煤炭和石油的数据然后分别计算这几年煤炭消费总量和石油消费总量，放在一张表里呈现",
    },
]

# 测试场景2: User message content 中的 cache_control
messages_user_cache = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Here is a long document that should be cached:\n\n" + "Lorem ipsum dolor sit amet. " * 100,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": "Based on the document above, please summarize the main points.",
            },
        ],
    },
]

# 测试场景3: 多轮对话中的 cache_control
messages_multi_turn_cache = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an expert programmer.",
                "cache_control": {"type": "ephemeral"},
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Here is my codebase:\n\n```python\ndef hello():\n    print('Hello World')\n```\n" * 50,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    },
    {
        "role": "assistant",
        "content": "I've reviewed your codebase. What would you like me to help with?",
    },
    {
        "role": "user",
        "content": "Please add error handling to the code.",
    },
]

# 测试场景4: Tools 中的 cache_control
tools_with_cache = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
        "cache_control": {"type": "ephemeral"},
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search the database for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "limit": {"type": "integer", "description": "Maximum results"},
                },
                "required": ["query"],
            },
        },
    },
]

messages_with_tools = [
    {
        "role": "system",
        "content": "You are a helpful assistant with access to tools.",
    },
    {
        "role": "user",
        "content": "What's the weather like in Beijing?",
    },
]

messages_tools_cache_regression = [
    {
        "role": "system",
        "content": "You are a careful assistant. Do not call tools unless required.",
    },
    {
        "role": "user",
        "content": "Reply with the word READY only.",
    },
]


def _build_prompt_cache_regression_messages() -> list[dict[str, Any]]:
    run_id = f"prompt-cache-run-{time.time_ns()}"
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a concise assistant. Use the reference document below exactly as context.\n"
                        f"Run id: {run_id}\n\n"
                        f"{CACHE_REGRESSION_DOCUMENT}"
                    ),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": "Return only the first three section numbers mentioned in the reference document.",
        },
    ]


def _build_tools_cache_regression_payload() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_id = f"tools-cache-run-{time.time_ns()}"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "cached_reference_tool",
                "description": (
                    f"Run id: {run_id}. "
                    + " ".join(
                        [
                            "This tool description is intentionally verbose for prompt caching validation."
                        ]
                        * 120
                    )
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": " ".join(["Topic description for schema caching."] * 60),
                        },
                        "detail_level": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": " ".join(["Detail level guidance for schema caching."] * 40),
                        },
                    },
                    "required": ["topic"],
                },
            },
            "cache_control": {"type": "ephemeral"},
        }
    ]
    return messages_tools_cache_regression, tools


def _invoke_completion(client, formatted_messages: list[dict[str, Any]], *, stream: bool, tools=NOT_GIVEN) -> dict[str, Any]:
    start_time = time.perf_counter()

    if not stream:
        response = client.create_completion(
            messages=formatted_messages,
            stream=False,
            tools=tools,
            skip_cutoff=True,
            max_tokens=1000,
        )
        content = response.content or ""
        usage = response.usage
    else:
        response = client.create_completion(
            messages=formatted_messages,
            stream=True,
            tools=tools,
            skip_cutoff=True,
            max_tokens=1000,
        )
        chunks: list[str] = []
        usage = None
        for chunk in response:
            if chunk.content:
                chunks.append(chunk.content)
            if chunk.usage:
                usage = chunk.usage
        content = "".join(chunks)

    elapsed_s = time.perf_counter() - start_time
    return {
        "content": content,
        "usage": usage,
        "elapsed_s": elapsed_s,
    }


def _cached_tokens(usage: Usage | None) -> int:
    if usage is None or usage.prompt_tokens_details is None:
        return 0
    return usage.prompt_tokens_details.cached_tokens or 0


def _cache_creation_tokens(usage: Usage | None) -> int:
    if usage is None or usage.cache_creation_tokens is None:
        return 0
    return usage.cache_creation_tokens


def _assert_repeated_cache_hits(
    name: str,
    *,
    messages: list,
    model: str,
    stream: bool,
    tools=NOT_GIVEN,
) -> dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(f"Testing: {name}")
    print(f"{'=' * 60}")

    client = create_chat_client(BackendType.Anthropic, model=model, stream=stream)
    formatted_messages = format_messages(messages, backend=BackendType.Anthropic)

    first = _invoke_completion(client, formatted_messages, stream=stream, tools=tools)
    second = _invoke_completion(client, formatted_messages, stream=stream, tools=tools)

    first_creation_tokens = _cache_creation_tokens(first["usage"])
    second_cached_tokens = _cached_tokens(second["usage"])

    print(f"First call: elapsed={first['elapsed_s']:.2f}s usage={first['usage']}")
    print(f"Second call: elapsed={second['elapsed_s']:.2f}s usage={second['usage']}")

    assert first["content"], f"Expected first response content, got {first['content']!r}"
    assert second["content"], f"Expected second response content, got {second['content']!r}"
    assert first_creation_tokens > 0, f"Expected first request to create cache tokens, got usage={first['usage']}"
    assert second_cached_tokens > 0, f"Expected second request to reuse cached tokens, got usage={second['usage']}"
    assert second_cached_tokens == first_creation_tokens, (
        "Expected second request cached tokens to match the first request cache creation tokens, "
        f"got cached_tokens={second_cached_tokens}, cache_creation_tokens={first_creation_tokens}"
    )

    return {
        "first": first,
        "second": second,
    }


def run_cache_scenario(name: str, messages: list, tools=NOT_GIVEN, model: str = DEFAULT_MODEL, stream: bool = False) -> dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(f"Testing: {name}")
    print(f"{'=' * 60}")

    client = create_chat_client(BackendType.Anthropic, model=model, stream=stream)
    formatted_messages = format_messages(messages, backend=BackendType.Anthropic)

    print("\nFormatted messages preview:")
    for i, msg in enumerate(formatted_messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            for j, block in enumerate(content):
                if isinstance(block, dict):
                    block_type = block.get("type", "unknown")
                    has_cache = "cache_control" in block
                    print(f"  Message {i} ({role}), Block {j}: type={block_type}, has_cache_control={has_cache}")
        else:
            content_preview = content[:50] + "..." if len(str(content)) > 50 else content
            print(f"  Message {i} ({role}): {content_preview}")

    if tools is not NOT_GIVEN:
        print("\nTools preview:")
        for i, tool in enumerate(tools):
            has_cache = "cache_control" in tool
            func_name = tool.get("function", {}).get("name", "unknown")
            print(f"  Tool {i}: name={func_name}, has_cache_control={has_cache}")

    result = _invoke_completion(client, formatted_messages, stream=stream, tools=tools)
    content = result["content"]
    usage = result["usage"]

    preview = content[:200] + "..." if len(content) > 200 else content
    print(f"\nResponse content: {preview}")
    print(f"\nUsage: {usage}")
    if usage is not None and usage.prompt_tokens_details is not None:
        print(f"Cache details: {usage.prompt_tokens_details}")
    print(f"\nTime elapsed: {result['elapsed_s']:.2f} seconds")

    return result


def run_cache_regression(model: str = DEFAULT_MODEL, stream: bool = False) -> dict[str, Any]:
    return _assert_repeated_cache_hits(
        "Repeated Request Cache Regression",
        messages=_build_prompt_cache_regression_messages(),
        model=model,
        stream=stream,
    )


def run_tools_cache_regression(model: str = DEFAULT_MODEL, stream: bool = False) -> dict[str, Any]:
    messages, tools = _build_tools_cache_regression_payload()
    return _assert_repeated_cache_hits(
        "Repeated Tools Cache Regression",
        messages=messages,
        tools=tools,
        model=model,
        stream=stream,
    )


def run_all_tests(model: str = DEFAULT_MODEL, stream: bool = False) -> None:
    print(f"\n{'#' * 60}")
    print("Running Prompt Cache Tests")
    print(f"Model: {model}, Stream: {stream}")
    print(f"{'#' * 60}")

    run_cache_scenario(
        name="System Message Cache (list format)",
        messages=messages_system_cache,
        model=model,
        stream=stream,
    )
    run_cache_scenario(
        name="User Message Content Cache",
        messages=messages_user_cache,
        model=model,
        stream=stream,
    )
    run_cache_scenario(
        name="Multi-turn Conversation Cache",
        messages=messages_multi_turn_cache,
        model=model,
        stream=stream,
    )
    run_cache_scenario(
        name="Tools Cache Control",
        messages=messages_with_tools,
        tools=tools_with_cache,
        model=model,
        stream=stream,
    )
    run_cache_regression(model=model, stream=stream)
    run_tools_cache_regression(model=model, stream=stream)

    print(f"\n{'#' * 60}")
    print("All tests completed!")
    print(f"{'#' * 60}")


def test_prompt_cache_regression() -> None:
    backend, model = resolve_backend_model(BackendType.Anthropic, DEFAULT_MODEL, presets=PRESETS)
    if backend != BackendType.Anthropic:
        raise ValueError("test_claude_prompt_cache only supports BackendType.Anthropic.")
    run_cache_regression(model=model, stream=False)


def test_tools_cache_regression() -> None:
    backend, model = resolve_backend_model(BackendType.Anthropic, DEFAULT_MODEL, presets=PRESETS)
    if backend != BackendType.Anthropic:
        raise ValueError("test_claude_prompt_cache only supports BackendType.Anthropic.")
    run_tools_cache_regression(model=model, stream=False)


def main() -> None:
    backend, model = resolve_backend_model(BackendType.Anthropic, DEFAULT_MODEL, presets=PRESETS)
    if backend != BackendType.Anthropic:
        raise ValueError("test_claude_prompt_cache only supports BackendType.Anthropic.")
    run_all_tests(model=model, stream=False)


if __name__ == "__main__":
    main()
