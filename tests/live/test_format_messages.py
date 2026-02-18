# @Author: Bi Ying
# @Date:   2024-07-27 11:51:28
import time
from pathlib import Path

from vv_llm.settings import settings
from vv_llm.types.enums import BackendType
from vv_llm.chat_clients import create_chat_client
from vv_llm.chat_clients.utils import format_messages

from live_common import load_live_settings, resolve_backend_model

load_live_settings(settings)
image = Path(__file__).parent / Path("./cat.png")

vectorvein_messages = [
    {
        "author_type": "U",
        "content_type": "TXT",
        "content": {
            "text": "描述这张图片。",
        },
        "attachments": [str(image)],
    }
]

if __name__ == "__main__":
    presets = {
        "gemini-3-flash-preview": (BackendType.Gemini, "gemini-3-flash-preview"),
        "gpt-4o": (BackendType.OpenAI, "gpt-4o"),
        "moonshot-v1-8k": (BackendType.Moonshot, "moonshot-v1-8k"),
        "claude-3.5-sonnet": (BackendType.Anthropic, "claude-3-5-sonnet-20240620"),
    }
    backend, model = resolve_backend_model(BackendType.Gemini, "gemini-3-flash-preview", presets=presets)

    messages = format_messages(vectorvein_messages, backend=backend, native_multimodal=True)

    start_time = time.perf_counter()
    client = create_chat_client(backend, model=model, stream=False)
    response = client.create_completion(messages=messages)
    print(response)
    end_time = time.perf_counter()
    print(f"Stream time elapsed: {end_time - start_time} seconds")
