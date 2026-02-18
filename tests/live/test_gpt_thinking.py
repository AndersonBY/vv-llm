# @Author: Bi Ying
# @Date:   2024-07-27 11:51:28
import asyncio
import time

from vv_llm.settings import settings
from vv_llm.chat_clients import (
    BackendType,
    format_messages,
    create_chat_client,
    create_async_chat_client,
)
from vv_llm.types.llm_parameters import NOT_GIVEN, ToolParam, ChatCompletionMessageParam

from live_common import load_live_settings, resolve_bool, resolve_backend_model


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

messages_for_multiple_calls = [
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
]


system = "You are a helpful assistant with access to tools."
messages_for_tools_simple: list[ChatCompletionMessageParam] = [{"role": "system", "content": system}]
messages_for_tools_simple.extend(
    [
        {
            "role": "user",
            # "content": "总结一下这个视频内容 https://www.bilibili.com/video/BV17C41187P2",
            "content": "画一个长毛橘猫图片",
            # "content": "Draw a picture of a long-haired orange cat",
            # "content": "周大福最新股价多少？",
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


def test_sync(model, stream: bool = False, use_tool: bool = False):
    client = create_chat_client(BackendType.OpenAI, model=model)
    if use_tool:
        messages = messages_for_tools_simple
        tools_params = tools_simple
    else:
        messages = messages_simple
        tools_params = NOT_GIVEN

    if not stream:
        response = client.create_completion(
            messages=format_messages(messages, backend=BackendType.OpenAI),
            stream=False,
            tools=tools_params,
            skip_cutoff=True,
            max_tokens=16000,
            reasoning_effort=reasoning_effort,
        )
        print(response)
    else:
        response = client.create_stream(
            messages=format_messages(messages, backend=BackendType.OpenAI),
            tools=tools_params,
            skip_cutoff=True,
            max_tokens=16000,
            reasoning_effort=reasoning_effort,
        )
        for chunk in response:
            # if chunk.reasoning_content:
            #     print(chunk.reasoning_content, end="", flush=True)
            # else:
            #     print(chunk.content, end="", flush=True)
            print(chunk)
            print("=" * 20)


async def test_async(model, stream: bool = False, use_tool: bool = False):
    client = create_async_chat_client(BackendType.OpenAI, model=model)
    if use_tool:
        messages = messages_for_tools_simple
        tools_params = tools_simple
    else:
        messages = messages_simple
        tools_params = NOT_GIVEN

    if not stream:
        response = await client.create_completion(
            messages=format_messages(messages, backend=BackendType.OpenAI),
            stream=False,
            tools=tools_params,
            tool_choice="auto" if use_tool else NOT_GIVEN,
            skip_cutoff=True,
            max_tokens=16000,
            reasoning_effort=reasoning_effort,
        )
        print(response)
    else:
        response = await client.create_stream(
            messages=format_messages(messages, backend=BackendType.OpenAI),
            tools=tools_params,
            skip_cutoff=True,
            max_tokens=16000,
            reasoning_effort=reasoning_effort,
        )
        async for chunk in response:
            print(chunk)
            print("=" * 20)


if __name__ == "__main__":
    presets = {
        "gpt-5": (BackendType.OpenAI, "gpt-5"),
        "gpt-5-pro": (BackendType.OpenAI, "gpt-5-pro"),
        "o4-mini": (BackendType.OpenAI, "o4-mini"),
    }
    backend, model = resolve_backend_model(BackendType.OpenAI, "gpt-5", presets=presets)
    if backend != BackendType.OpenAI:
        raise ValueError("test_gpt_thinking only supports BackendType.OpenAI.")

    stream = resolve_bool("VV_LLM_STREAM", False)
    use_tool = resolve_bool("VV_LLM_USE_TOOL", False)
    use_async = resolve_bool("VV_LLM_USE_ASYNC", False)
    reasoning_effort = "high"

    start_time = time.perf_counter()
    if use_async:
        asyncio.run(test_async(model=model, stream=stream, use_tool=use_tool))
    else:
        test_sync(model=model, stream=stream, use_tool=use_tool)
    end_time = time.perf_counter()
    print(f"Time elapsed: {end_time - start_time} seconds")
