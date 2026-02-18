from __future__ import annotations

from copy import deepcopy

from vv_llm.chat_clients import create_chat_client
from vv_llm.types import BackendType

from live_common import resolve_backend_model, run_with_timer
from sample_settings import sample_settings

MESSAGES = [
    {
        "role": "user",
        "content": "恶魔猎手需要从福波斯传送至德摩斯。他带着他的宠物兔子、宠物地狱犬恶魔以及一个跟随的 UAC 科学家。恶魔猎手一次只能与他们中的一个一起传送。但如果他将兔子和地狱犬恶魔单独留下，兔子会吃掉地狱犬恶魔。如果他将地狱犬恶魔和科学家单独留下，地狱犬恶魔会吃掉科学家。恶魔猎手应该如何安全地将自己和所有同伴带到德摩斯？",
    }
]

PRESETS = {
    "deepseek-reasoner": (BackendType.DeepSeek, "deepseek-reasoner"),
    "deepseek-chat": (BackendType.DeepSeek, "deepseek-chat"),
}


def main() -> None:
    backend, model = resolve_backend_model(BackendType.DeepSeek, "deepseek-reasoner", presets=PRESETS)
    if backend != BackendType.DeepSeek:
        raise ValueError("test_custom_settings currently supports DeepSeek only. Use preset deepseek-reasoner/deepseek-chat.")

    local_settings = deepcopy(sample_settings)
    local_settings["deepseek"]["models"][model]["endpoints"] = ["deepseek-default"]

    def _run():
        client = create_chat_client(backend=backend, model=model, settings=local_settings)
        print(f"Endpoint: {client.endpoint}")
        response = client.create_stream(messages=MESSAGES, skip_cutoff=True)
        for chunk in response:
            print(chunk)
            print("=" * 20)

    run_with_timer("custom_settings", _run)


if __name__ == "__main__":
    main()
