from __future__ import annotations

from v_llm.chat_clients import create_chat_client
from v_llm.settings import settings
from v_llm.types.enums import BackendType

from live_common import load_live_settings, resolve_backend_model, run_with_timer

MESSAGES = [
    {
        "role": "user",
        "content": "节点名称是 FileLoader，FileLoader 节点连到 OCR 节点，使用 mermaid 语法表示流程图。```mermaid 方式包裹。",
    }
]

PRESETS = {
    "deepseek-reasoner": (BackendType.DeepSeek, "deepseek-reasoner"),
    "deepseek-chat": (BackendType.DeepSeek, "deepseek-chat"),
}


def main() -> None:
    load_live_settings(settings)
    backend, model = resolve_backend_model(BackendType.DeepSeek, "deepseek-reasoner", presets=PRESETS)

    def _run():
        client = create_chat_client(backend=backend, model=model, stream=False)
        response = client.create_completion(messages=MESSAGES, stop=["\n```"])
        print(response)

    run_with_timer("stop", _run)


if __name__ == "__main__":
    main()
