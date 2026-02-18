# @Author: Bi Ying
# @Date:   2024-07-27 11:51:28
import time

from v_llm.settings import settings
from v_llm.chat_clients import (
    BackendType,
    format_messages,
    create_chat_client,
)
from v_llm.types.llm_parameters import NOT_GIVEN

from live_common import load_live_settings, resolve_backend_model


load_live_settings(settings)

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


def test_cache_scenario(name: str, messages: list, tools=NOT_GIVEN, model: str = "claude-sonnet-4-5-20250929", stream: bool = False):
    """Test a specific cache scenario and print results"""
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
            content_preview = f"[{len(content)} blocks]"
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

    start_time = time.perf_counter()

    if not stream:
        response = client.create_completion(
            messages=formatted_messages,
            stream=False,
            tools=tools,
            skip_cutoff=True,
            max_tokens=1000,
        )
        print(f"\nResponse content: {response.content[:200]}..." if len(response.content) > 200 else f"\nResponse content: {response.content}")
        print(f"\nUsage: {response.usage}")
        if hasattr(response.usage, "prompt_tokens_details") and response.usage.prompt_tokens_details:
            print(f"Cache details: {response.usage.prompt_tokens_details}")
    else:
        response = client.create_completion(
            messages=formatted_messages,
            stream=True,
            tools=tools,
            skip_cutoff=True,
            max_tokens=1000,
        )
        content = ""
        usage = None
        for chunk in response:
            if chunk.content:
                content += chunk.content
            if chunk.usage:
                usage = chunk.usage
        print(f"\nResponse content: {content[:200]}..." if len(content) > 200 else f"\nResponse content: {content}")
        if usage:
            print(f"\nUsage: {usage}")

    end_time = time.perf_counter()
    print(f"\nTime elapsed: {end_time - start_time:.2f} seconds")

    return True


def run_all_tests(model: str = "claude-sonnet-4-5-20250929", stream: bool = False):
    """Run all cache control tests"""
    print(f"\n{'#' * 60}")
    print("Running Prompt Cache Tests")
    print(f"Model: {model}, Stream: {stream}")
    print(f"{'#' * 60}")

    # Test 1: System message cache
    test_cache_scenario(
        name="System Message Cache (list format)",
        messages=messages_system_cache,
        model=model,
        stream=stream,
    )

    # Test 2: User message cache
    test_cache_scenario(
        name="User Message Content Cache",
        messages=messages_user_cache,
        model=model,
        stream=stream,
    )

    # Test 3: Multi-turn cache
    test_cache_scenario(
        name="Multi-turn Conversation Cache",
        messages=messages_multi_turn_cache,
        model=model,
        stream=stream,
    )

    # Test 4: Tools cache
    test_cache_scenario(
        name="Tools Cache Control",
        messages=messages_with_tools,
        tools=tools_with_cache,
        model=model,
        stream=stream,
    )

    print(f"\n{'#' * 60}")
    print("All tests completed!")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    presets = {
        "sonnet-4.5": (BackendType.Anthropic, "claude-sonnet-4-5-20250929"),
        "sonnet-4.6": (BackendType.Anthropic, "claude-sonnet-4-6"),
        "haiku-3.5": (BackendType.Anthropic, "claude-3-5-haiku-20241022"),
    }
    backend, model = resolve_backend_model(BackendType.Anthropic, "claude-sonnet-4-5-20250929", presets=presets)
    if backend != BackendType.Anthropic:
        raise ValueError("test_claude_prompt_cache only supports BackendType.Anthropic.")
    stream = False

    # Run all tests
    run_all_tests(model=model, stream=stream)

    # Or run individual tests:
    # test_cache_scenario("System Cache", messages_system_cache, model=model)
    # test_cache_scenario("Tools Cache", messages_with_tools, tools=tools_with_cache, model=model)
