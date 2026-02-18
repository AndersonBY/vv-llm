# @Author: Bi Ying
# @Date:   2024-07-26 14:48:55
from ..types.enums import BackendType
from .openai_compatible_client import OpenAICompatibleChatClient, AsyncOpenAICompatibleChatClient


class LocalChatClient(OpenAICompatibleChatClient):
    DEFAULT_MODEL = ""
    BACKEND_NAME = BackendType.Local


class AsyncLocalChatClient(AsyncOpenAICompatibleChatClient):
    DEFAULT_MODEL = ""
    BACKEND_NAME = BackendType.Local
