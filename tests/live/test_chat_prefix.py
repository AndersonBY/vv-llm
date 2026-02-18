from __future__ import annotations

from v_llm.chat_clients import create_chat_client
from v_llm.settings import settings
from v_llm.types.enums import BackendType

from live_common import load_live_settings, resolve_backend_model, run_with_timer

MESSAGES = [
    {"role": "user", "content": "Please write quick sort code"},
    {"role": "assistant", "content": "```python\n", "prefix": True},
]

PRESETS = {
    "deepseek-chat": (BackendType.DeepSeek, "deepseek-chat"),
    "deepseek-reasoner": (BackendType.DeepSeek, "deepseek-reasoner"),
    "gpt-4o": (BackendType.OpenAI, "gpt-4o"),
}


def main() -> None:
    load_live_settings(settings)
    backend, model = resolve_backend_model(BackendType.DeepSeek, "deepseek-chat", presets=PRESETS)

    def _run():
        client = create_chat_client(backend=backend, model=model, stream=False)
        response = client.create_completion(messages=MESSAGES, stop=["\n```"])
        print(response)

    run_with_timer("chat_prefix", _run)


if __name__ == "__main__":
    main()
