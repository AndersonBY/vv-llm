from __future__ import annotations

from vv_llm.chat_clients import create_chat_client, format_messages
from vv_llm.settings import settings
from vv_llm.types.enums import BackendType
from vv_llm.types.exception import APIStatusError

from live_common import load_live_settings, resolve_backend_model, run_with_timer

MESSAGES = [
    {
        "role": "user",
        "content": 'List a few popular cookie recipes {"recipe_name": str} Return: list[Recipe]',
    },
]

PRESETS = {
    "deepseek-chat": (BackendType.DeepSeek, "deepseek-chat"),
    "qwen-72b": (BackendType.Qwen, "qwen2.5-72b-instruct"),
    "gpt-4o": (BackendType.OpenAI, "gpt-4o"),
}


def main() -> None:
    load_live_settings(settings)
    backend, model = resolve_backend_model(BackendType.DeepSeek, "deepseek-chat", presets=PRESETS)

    def _run():
        client = create_chat_client(backend=backend, model=model, stream=False)
        try:
            client.create_completion(
                messages=format_messages(MESSAGES, backend=backend),
                response_format={"type": "json_object"},
            )
        except APIStatusError as exc:
            print(f"type={type(exc).__name__}")
            print(f"status_code={exc.status_code}")
            print(f"message={exc.message}")
        else:
            print("Request succeeded; no APIStatusError raised.")

    run_with_timer("error_case", _run)


if __name__ == "__main__":
    main()
