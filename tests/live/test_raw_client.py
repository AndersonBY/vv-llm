from __future__ import annotations

from v_llm.chat_clients import create_chat_client
from v_llm.settings import settings
from v_llm.types.enums import BackendType

from live_common import load_live_settings, resolve_backend_model, run_with_timer

PRESETS = {
    "anthropic": (BackendType.Anthropic, "claude-sonnet-4-6"),
    "deepseek": (BackendType.DeepSeek, "deepseek-chat"),
    "openai": (BackendType.OpenAI, "gpt-4o"),
}


def main() -> None:
    load_live_settings(settings)
    backend, model = resolve_backend_model(BackendType.Anthropic, "claude-sonnet-4-6", presets=PRESETS)

    def _run():
        client = create_chat_client(backend=backend, model=model, stream=False)
        print(client.raw_client)

    run_with_timer("raw_client", _run)


if __name__ == "__main__":
    main()
