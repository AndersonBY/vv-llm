# @Author: Bi Ying
# @Date:   2024-07-26 14:48:55
from ..types.enums import BackendType
from ..types.defaults import DEEPSEEK_DEFAULT_MODEL
from .openai_compatible_client import OpenAICompatibleChatClient, AsyncOpenAICompatibleChatClient


class DeepSeekChatClient(OpenAICompatibleChatClient):
    DEFAULT_MODEL = DEEPSEEK_DEFAULT_MODEL
    BACKEND_NAME = BackendType.DeepSeek


class AsyncDeepSeekChatClient(AsyncOpenAICompatibleChatClient):
    DEFAULT_MODEL = DEEPSEEK_DEFAULT_MODEL
    BACKEND_NAME = BackendType.DeepSeek
