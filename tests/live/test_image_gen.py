# @Author: Bi Ying
# @Date:   2024-07-27 11:51:28
import time
from pathlib import Path

from v_llm.settings import settings
from v_llm.types.enums import BackendType
from v_llm.types.llm_parameters import VectorVeinMessage, ChatCompletionMessageParam
from v_llm.chat_clients.utils import format_messages
from v_llm.utilities.media_processing import ImageProcessor
from v_llm.chat_clients import create_chat_client

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
            {
                "type": "text",
                "text": """画一个飞轮的图:
将合作伙伴、普通用户的动力都调动起来，形成多赢局面：
1）Servers：参考我们的campaign blog，策划campaign，既能强化各社区自身的凝聚力，促进老用户活跃，同时在discordhunt被feature，campaign吸引新用户参与，帮助社区拉新->社区获得新用户和知名度提升
2）Discordhunt：被通知到的servers->愿意参与的servers，吸引被通知/想参与的servers，吸引流量->Discordhunt平台获得内容和流量的冷启动
3）用户：所有关注到活动的用户，既可以upvote、评论，也可以到各个群参加活动，获得优惠、奖品等福利->用户获得奖励和荣誉感

直接画出来，不需要解释。""",
            },
            # {
            #     "type": "image_url",
            #     "image_url": {
            #         "url": image_processor.data_url,
            #     },
            # },
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
        "gemini-3-pro-image-preview": (BackendType.Gemini, "gemini-3-pro-image-preview"),
        "claude-sonnet-4": (BackendType.Anthropic, "claude-sonnet-4-20250514"),
        "qwen3-vl-plus": (BackendType.Qwen, "qwen3-vl-plus"),
        "glm-4.6v-flash": (BackendType.ZhiPuAI, "glm-4.6v-flash"),
    }
    backend, model = resolve_backend_model(BackendType.Gemini, "gemini-3-pro-image-preview", presets=presets)
    use_vectorvein_messages = resolve_bool("VLLM_USE_VECTORVEIN_MESSAGES", False)

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

    response = client.create_completion(
        messages=formatted_messages,
        skip_cutoff=True,
        max_completion_tokens=16000,
    )
    print(response)
    end_time = time.perf_counter()
    print(f"Stream time elapsed: {end_time - start_time} seconds")
