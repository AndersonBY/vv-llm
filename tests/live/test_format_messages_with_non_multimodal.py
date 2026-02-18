from __future__ import annotations

from v_llm.chat_clients import create_chat_client
from v_llm.chat_clients.utils import format_messages
from v_llm.settings import settings
from v_llm.types.enums import BackendType

from live_common import load_live_settings, resolve_backend_model, run_with_timer

MESSAGES = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://vector-vein-cdn.oss-cn-hangzhou.aliyuncs.com/documents/resources/images/nodes/tools-text-search.jpg",
                },
            },
            {"type": "text", "text": "你能告诉我这个图片的域名是什么吗？"},
        ],
    }
]

PRESETS = {
    "deepseek-reasoner": (BackendType.DeepSeek, "deepseek-reasoner"),
    "qwen-72b": (BackendType.Qwen, "qwen2.5-72b-instruct"),
    "gpt-4o": (BackendType.OpenAI, "gpt-4o"),
}


def main() -> None:
    load_live_settings(settings)
    backend, model = resolve_backend_model(BackendType.DeepSeek, "deepseek-reasoner", presets=PRESETS)

    def _run():
        formatted = format_messages(MESSAGES, backend=backend)
        print(formatted)
        client = create_chat_client(backend, model=model, stream=False)
        print(client.create_completion(messages=formatted))

    run_with_timer("format_messages_non_multimodal", _run)


if __name__ == "__main__":
    main()
