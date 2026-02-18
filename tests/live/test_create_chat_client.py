# @Author: Bi Ying
# @Date:   2024-07-27 11:51:28
import asyncio
import time

from v_llm.settings import settings
from v_llm.chat_clients import (
    BackendType,
    format_messages,
    create_chat_client,
    create_async_chat_client,
)
from v_llm.types.llm_parameters import NOT_GIVEN, ToolParam
from openai.types.chat import ChatCompletionMessageParam

from live_common import load_live_settings, resolve_backend_model, resolve_bool

load_live_settings(settings)

tools_for_multiple_calls: list[ToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
            },
        },
    }
]

messages_for_multiple_calls: list[ChatCompletionMessageParam] = [
    {
        "role": "system",
        "content": "You are a helpful assistant that can access external functions. The responses from these function calls will be appended to this dialogue. Please provide responses based on the information from these function calls.",
    },
    {"role": "user", "content": "What is the current temperature of New York, San Francisco and Chicago?"},
]


tools_simple: list[ToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "video2mindmap_speech_recognition",
            "description": "输入B站视频网址后可获得\n- 语音识别的文字结果\n- 根据语音识别结果得到的内容总结",
            "parameters": {
                "type": "object",
                "required": ["url_or_bvid"],
                "properties": {"url_or_bvid": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dall_e_image_generation",
            "description": "输入一段文字，Dall-E 根据文字内容生成一张图片。",
            "parameters": {"type": "object", "required": ["prompt"], "properties": {"prompt": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gpt_vision_url_image_analysis",
            "description": "输入图像链接以及一段文字，让 GPT-Vision 根据文字和图像生成回答文字。",
            "parameters": {
                "type": "object",
                "required": ["urls", "text_prompt"],
                "properties": {
                    "urls": {"type": "string"},
                    "text_prompt": {"type": "string", "description": "文字提示"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bing_search",
            "description": "输入搜索关键词，返回Bing搜索结果",
            "parameters": {
                "type": "object",
                "required": ["search_text"],
                "properties": {
                    "search_text": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_xhs",
            "description": "指定搜索关键词给出小红书笔记搜索结果",
            "parameters": {
                "type": "object",
                "required": ["keywords"],
                "properties": {
                    "page": {"type": "integer", "description": "页码，默认为1", "default": 1},
                    "keywords": {"type": "string", "description": "搜索关键词"},
                    "note_type": {"type": "string", "description": "笔记类型 (0: 全部, 1: 视频, 2: 图文)", "default": "0"},
                    "sort_type": {"type": "string", "description": "排序方式 (general, time_descending, popularity_descending)", "default": "general"},
                },
            },
        },
    },
]


system = "You are a helpful assistant with access to tools."
messages_for_tools_simple: list[ChatCompletionMessageParam] = [{"role": "system", "content": system}]
messages_for_tools_simple.extend(
    [
        {
            "role": "user",
            # "content": "总结一下这个视频内容 https://www.bilibili.com/video/BV17C41187P2",
            # "content": "画一个长毛橘猫图片",
            # "content": "Draw a picture of a long-haired orange cat",
            # "content": "周大福最新股价多少？",
            "content": "搜一下小红书上关于 AI绘画 和 AI Agent 的笔记，直接搜，不要提问。",
        },
    ]
)

messages_simple: list[ChatCompletionMessageParam] = [
    {
        "role": "system",
        "content": "As an expert in SQLite, you are expected to utilize your knowledge to craft SQL queries that are in strict adherence with SQLite syntax standards when responding to inquiries.",
    },
    {
        "role": "user",
        "content": "The table structure is as follows:\n```sql\nCREATE TABLE 历年能源消费构成 (\n    年份        INTEGER,\n    煤炭        INTEGER,\n    石油        INTEGER,\n    天然气       INTEGER,\n    一次电力及其他能源 INTEGER\n);\n\n```\n\nPlease write the SQL to answer the question: `筛出2005年到2010年之间煤炭和石油的数据然后分别计算这几年煤炭消费总量和石油消费总量，放在一张表里呈现`\nDo not explain.",
    },
]

messages_simple: list[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": "恶魔猎手需要从福波斯传送至德摩斯。他带着他的宠物兔子、宠物地狱犬恶魔以及一个跟随的 UAC 科学家。恶魔猎手一次只能与他们中的一个一起传送。但如果他将兔子和地狱犬恶魔单独留下，兔子会吃掉地狱犬恶魔。如果他将地狱犬恶魔和科学家单独留下，地狱犬恶魔会吃掉科学家。恶魔猎手应该如何安全地将自己和所有同伴带到德摩斯？",
    },
]


def test_sync(backend, model, stream: bool = False, use_tool: bool = False):
    client = create_chat_client(backend, model=model, stream=stream)
    if model.startswith("gemini-2.5"):
        extra_body = {
            "extra_body": {
                "google": {
                    "thinking_config": {
                        "thinkingBudget": -1,
                        "include_thoughts": True,
                    }
                }
            }
        }
    elif model.startswith("gemini-3"):
        extra_body = {
            "google": {
                "thinking_config": {
                    "thinkingLevel": "high",
                    "include_thoughts": True,
                }
            }
        }
    elif model.startswith("glm-5"):
        extra_body = {"tool_stream": True}
    else:
        extra_body = None
    if use_tool:
        messages = messages_for_tools_simple
        tools_params = tools_simple
    else:
        messages = messages_simple
        messages = [{"role": "user", "content": "How many 'r's are in the word strawberry?"}]
        tools_params = NOT_GIVEN

    if not stream:
        response = client.create_completion(
            messages=format_messages(messages, backend=backend),
            stream=False,
            tools=tools_params,
            # temperature=1,
            skip_cutoff=True,
            timeout=30,
            max_tokens=8192,
            extra_body=extra_body,
        )
        print(response)
    else:
        response = client.create_stream(
            messages=format_messages(messages, backend=backend),
            tools=tools_params,
            # temperature=1,
            stream_options={"include_usage": True},
            skip_cutoff=True,
            max_tokens=2048,
            timeout=30,
            # extra_body={"enable_thinking": False},
            extra_body=extra_body,
        )
        start_content = False
        for chunk in response:
            if chunk.reasoning_content:
                print(chunk.reasoning_content, end="")
            else:
                if not start_content:
                    start_content = True
                    print("\n=== Content Start ===\n")
                print(chunk.content, end="")
            if use_tool:
                print(chunk.tool_calls)
            if chunk.usage:
                print(f"Usage: {chunk.usage}")
                print("=" * 20)


async def test_async(backend, model, stream: bool = False, use_tool: bool = False):
    client = create_async_chat_client(backend, model=model)
    if use_tool:
        messages = messages_for_tools_simple
        tools_params = tools_simple
    else:
        messages = messages_simple
        tools_params = NOT_GIVEN

    if not stream:
        response = await client.create_completion(
            messages=format_messages(messages, backend=backend),
            stream=False,
            tools=tools_params,
            tool_choice="auto" if use_tool else NOT_GIVEN,
            skip_cutoff=True,
            timeout=30,
            # temperature=1,
        )
        print(response)
    else:
        response = await client.create_stream(
            messages=format_messages(messages, backend=backend),
            tools=tools_params,
            skip_cutoff=True,
            timeout=30,
            # temperature=1,
        )
        async for chunk in response:
            print(chunk)
            print("=" * 20)


if __name__ == "__main__":
    presets = {
        "claude-sonnet-4.6": (BackendType.Anthropic, "claude-sonnet-4-6"),
        "deepseek-chat": (BackendType.DeepSeek, "deepseek-chat"),
        "deepseek-reasoner": (BackendType.DeepSeek, "deepseek-reasoner"),
        "gemini-2.5-pro": (BackendType.Gemini, "gemini-2.5-pro"),
        "gemini-3-pro-preview": (BackendType.Gemini, "gemini-3-pro-preview"),
        "kimi-k2.5": (BackendType.Moonshot, "kimi-k2.5"),
        "moonshot-v1-8k": (BackendType.Moonshot, "moonshot-v1-8k"),
        "gpt-4o": (BackendType.OpenAI, "gpt-4o"),
        "gpt-5": (BackendType.OpenAI, "gpt-5"),
        "gpt-5-pro": (BackendType.OpenAI, "gpt-5-pro"),
        "minimax-m2.5": (BackendType.MiniMax, "MiniMax-M2.5"),
        "yi-large-fc": (BackendType.Yi, "yi-large-fc"),
        "qwen3-235b": (BackendType.Qwen, "qwen3-235b-a22b"),
        "glm-5": (BackendType.ZhiPuAI, "glm-5"),
        "grok-4": (BackendType.XAI, "grok-4"),
    }
    backend, model = resolve_backend_model(BackendType.Anthropic, "claude-sonnet-4-6", presets=presets)

    stream = resolve_bool("VLLM_STREAM", True)
    use_tool = resolve_bool("VLLM_USE_TOOL", True)
    use_async = resolve_bool("VLLM_USE_ASYNC", False)

    start_time = time.perf_counter()
    if use_async:
        asyncio.run(test_async(backend=backend, model=model, stream=stream, use_tool=use_tool))
    else:
        test_sync(backend=backend, model=model, stream=stream, use_tool=use_tool)
    end_time = time.perf_counter()
    print(f"Time elapsed: {end_time - start_time} seconds")
