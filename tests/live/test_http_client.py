from __future__ import annotations

import httpx

from v_llm.chat_clients import create_chat_client
from v_llm.settings import settings
from v_llm.types.enums import BackendType

from live_common import load_live_settings, resolve_backend_model, run_with_timer

MESSAGES = [{"role": "user", "content": "Please write quick sort code"}]

PRESETS = {
    "deepseek": (BackendType.DeepSeek, "deepseek-chat"),
    "openai": (BackendType.OpenAI, "gpt-4o"),
}


def main() -> None:
    load_live_settings(settings)
    backend, model = resolve_backend_model(BackendType.DeepSeek, "deepseek-chat", presets=PRESETS)

    def _run():
        with httpx.Client() as http_client:
            client = create_chat_client(backend=backend, model=model, stream=False, http_client=http_client)
            response = client.create_completion(messages=MESSAGES)
            print(response)

    run_with_timer("http_client", _run)


if __name__ == "__main__":
    main()
