from __future__ import annotations

from vv_llm.chat_clients import create_chat_client, format_messages
from vv_llm.settings import settings
from vv_llm.types.enums import BackendType

from live_common import load_live_settings, resolve_backend_model, run_with_timer

MESSAGES = [
    {
        "role": "user",
        "content": 'List a few popular cookie recipes using this JSON schema: Recipe = {"recipe_name": str} Return: list[Recipe]',
    }
]

PRESETS = {
    "qwen-72b": (BackendType.Qwen, "qwen2.5-72b-instruct"),
    "gpt-4o": (BackendType.OpenAI, "gpt-4o"),
    "kimi-k2.5": (BackendType.Moonshot, "kimi-k2.5"),
}


def main() -> None:
    load_live_settings(settings)
    backend, model = resolve_backend_model(BackendType.Moonshot, "kimi-k2.5", presets=PRESETS)

    def _run():
        client = create_chat_client(backend=backend, model=model, stream=False)
        response = client.create_completion(
            messages=format_messages(MESSAGES, backend=backend),
            response_format={"type": "json_object"},
            skip_cutoff=True,
        )
        print(response)

    run_with_timer("skip_cutoff", _run)


if __name__ == "__main__":
    main()
