import asyncio

from vv_llm import ChatRequest, ChatRequestOptions

from common import load_client


async def main() -> None:
    client = load_client(asynchronous=True)
    stream = await client.create(
        ChatRequest(
            model=client.model,
            messages=[{"role": "user", "content": "Write a four-line poem."}],
            stream=True,
            options=ChatRequestOptions(max_tokens=128),
        )
    )
    async for chunk in stream:
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
