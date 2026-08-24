import json

from vv_llm import ChatRequest


def main() -> None:
    canonical = {
        "model": "example-model",
        "messages": [{"role": "user", "content": "Hello from the contract codec."}],
        "options": {"stream": False, "max_tokens": 64},
    }
    request = ChatRequest.from_contract(canonical)
    print(json.dumps(request.to_contract(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
