from vv_llm import ChatRequest, ChatRequestOptions

from common import load_client


def main() -> None:
    client = load_client()
    stream = client.create(
        ChatRequest(
            model=client.model,
            messages=[{"role": "user", "content": "Write a short haiku."}],
            stream=True,
            options=ChatRequestOptions(max_tokens=128),
        )
    )
    for chunk in stream:
        if chunk.reasoning_content:
            print(chunk.reasoning_content, end="", flush=True)
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
