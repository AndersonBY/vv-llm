# @Author: Bi Ying
# @Date:   2024-07-26 14:48:55
from ..types.enums import BackendType
from ..types.defaults import QWEN_DEFAULT_MODEL
from .openai_compatible_client import OpenAICompatibleChatClient, AsyncOpenAICompatibleChatClient


class QwenChatClient(OpenAICompatibleChatClient):
    DEFAULT_MODEL = QWEN_DEFAULT_MODEL
    BACKEND_NAME = BackendType.Qwen


class AsyncQwenChatClient(AsyncOpenAICompatibleChatClient):
    DEFAULT_MODEL = QWEN_DEFAULT_MODEL
    BACKEND_NAME = BackendType.Qwen
