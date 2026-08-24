from vv_llm import ChatRequest, ChatRequestOptions, ThinkingPreference

from common import load_client


def main() -> None:
    client = load_client()
    response = client.create(
        ChatRequest(
            model=client.model,
            messages=[{"role": "user", "content": "Compute 37 * 19 and answer briefly."}],
            options=ChatRequestOptions(
                thinking=ThinkingPreference.enabled(),
                max_tokens=256,
            ),
        )
    )
    print("reasoning:", response.reasoning_content or "")
    print("answer:", response.content or "")


if __name__ == "__main__":
    main()
