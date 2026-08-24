import os

from vv_llm import ChatRequest

from common import load_client


def main() -> None:
    client = load_client()
    image_url = os.environ.get("VV_LLM_IMAGE_URL", "https://example.com/image.png")
    request = ChatRequest.from_contract(
        {
            "model": client.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe these images.", "cache_control": {"type": "ephemeral"}},
                        {"type": "image_url", "url": image_url, "detail": "low", "cache_control": {"type": "ephemeral"}},
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                    ],
                }
            ],
            "options": {"max_tokens": 256},
        }
    )
    response = client.create(request)
    print(response.content or "")


if __name__ == "__main__":
    main()
