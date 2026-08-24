from vv_llm import ChatRequest

from common import load_client


def main() -> None:
    client = load_client()
    request = ChatRequest.from_contract(
        {
            "model": client.model,
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "options": {"max_tokens": 256},
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        }
    )
    response = client.create(request)
    print(response.content or "")
    for tool_call in response.tool_calls or []:
        print(tool_call)


if __name__ == "__main__":
    main()
