# @Author: Bi Ying
# @Date:   2024-07-27 11:51:28
from pathlib import Path

from vv_llm.settings import settings
from vv_llm.chat_clients import (
    BackendType,
    get_token_counts,
    get_message_token_counts,
)
from vv_llm.utilities.media_processing import ImageProcessor

from live_common import load_live_settings, resolve_backend_model


load_live_settings(settings)


if __name__ == "__main__":
    presets = {
        "claude-sonnet-4.6": (BackendType.Anthropic, "claude-sonnet-4-6"),
        "kimi-k2": (BackendType.Moonshot, "kimi-k2-0711-preview"),
        "gpt-4o": (BackendType.OpenAI, "gpt-4o"),
        "gemini-2.5-pro": (BackendType.Gemini, "gemini-2.5-pro"),
        "qwen2-72b": (BackendType.Qwen, "qwen2-72b-instruct"),
    }
    _, model = resolve_backend_model(BackendType.Anthropic, "claude-sonnet-4-6", presets=presets)

    tokens = get_token_counts("hello 我是毕老师", model, use_token_server_first=False)
    print(tokens)

    image = Path(__file__).parent / "cat.png"
    image_processor = ImageProcessor(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "描述这张图片。"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_processor.mime_type,
                        "data": image_processor.base64_image,
                    },
                },
                {"type": "text", "text": "描述这张图片。"},
                {"type": "text", "text": "描述这张图片。"},
                {"type": "text", "text": "描述这张图片。"},
            ],
        }
    ]
    tokens = get_message_token_counts(messages=messages, model=model)
    print(tokens)
