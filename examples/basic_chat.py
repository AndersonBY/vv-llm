from vv_llm import ChatRequest, ChatRequestOptions

from common import load_client


def main() -> None:
    client = load_client()
    response = client.create(
        ChatRequest(
            model=client.model,
            messages=[{"role": "user", "content": "Explain RAG in one sentence."}],
            options=ChatRequestOptions(max_tokens=128),
        )
    )
    print(response.content or "")


if __name__ == "__main__":
    main()
