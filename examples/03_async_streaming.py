import asyncio

from vv_llm import ChatRequest, ChatRequestOptions, ThinkingPreference

from common import load_deepseek_client


async def main() -> None:
    client = load_deepseek_client(asynchronous=True)
    stream = await client.create(
        ChatRequest(
            messages=[{"role": "user", "content": "Write a four-line poem."}],
            stream=True,
            options=ChatRequestOptions(
                thinking=ThinkingPreference.disabled(),
                max_tokens=256,
            ),
        )
    )

    async for chunk in stream:
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


asyncio.run(main())
