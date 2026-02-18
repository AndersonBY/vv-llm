# @Author: Bi Ying
# @Date:   2024-07-27 11:51:28
import time
from pathlib import Path

from vv_llm.settings import settings
from vv_llm.types.enums import BackendType
from vv_llm.types.llm_parameters import VectorVeinMessage, ChatCompletionMessageParam
from vv_llm.chat_clients.utils import format_messages
from vv_llm.utilities.media_processing import ImageProcessor
from vv_llm.chat_clients import create_chat_client, get_message_token_counts

from live_common import load_live_settings, resolve_backend_model, resolve_bool

load_live_settings(settings)
image = Path(__file__).parent / Path("./cat.png")
image_processor = ImageProcessor(image)

anthropic_messages = [
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
        ],
    }
]

openai_messages: list[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "描述这张图片。"},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_processor.data_url,
                },
            },
        ],
    }
]

vectorvein_messages: list[VectorVeinMessage] = [
    {
        "author_type": "U",
        "content_type": "TXT",
        "content": {
            "text": "描述这张图片。",
        },
        "attachments": [str(image)],
    }
]

# openai_video_messages: list[ChatCompletionMessageParam] = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": "描述这个视频内容。"},
#             {
#                 "type": "video_url",
#                 "video_url": {
#                     "url": "https://video.twimg.com/amplify_video/1940347508267384832/vid/avc1/1280x720/8A1YvT7nGeXmpHKa.mp4?tag=21",
#                 },
#             },
#         ],
#     }
# ]


if __name__ == "__main__":
    presets = {
        "claude-sonnet-4.6": (BackendType.Anthropic, "claude-sonnet-4-6"),
        "qwen3-vl-flash": (BackendType.Qwen, "qwen3-vl-flash"),
        "qwen3-vl-plus": (BackendType.Qwen, "qwen3-vl-plus"),
        "glm-4.6v-flash": (BackendType.ZhiPuAI, "glm-4.6v-flash"),
    }
    backend, model = resolve_backend_model(BackendType.ZhiPuAI, "glm-4.6v-flash", presets=presets)
    use_vectorvein_messages = resolve_bool("VV_LLM_USE_VECTORVEIN_MESSAGES", False)

    start_time = time.perf_counter()
    client = create_chat_client(backend, model=model, stream=False)
    native_multimodal = client.backend_settings.models[model].native_multimodal
    print(f"Is native multimodal: {native_multimodal}")

    source_messages = vectorvein_messages if use_vectorvein_messages else openai_messages
    formatted_messages = format_messages(
        messages=source_messages,
        backend=backend,
        native_multimodal=native_multimodal,
    )

    response = client.create_stream(messages=formatted_messages)
    for chunk in response:
        print(chunk)
        print("=" * 20)
    end_time = time.perf_counter()
    print(f"Stream time elapsed: {end_time - start_time} seconds")

    tokens = get_message_token_counts(messages=openai_messages, model=model)
    print(f"Predict tokens: {tokens}")
