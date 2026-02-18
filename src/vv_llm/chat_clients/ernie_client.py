# @Author: Bi Ying
# @Date:   2024-07-26 14:48:55
from ..types.enums import BackendType
from ..types.defaults import ERNIE_DEFAULT_MODEL
from .openai_compatible_client import OpenAICompatibleChatClient, AsyncOpenAICompatibleChatClient


class ErnieChatClient(OpenAICompatibleChatClient):
    DEFAULT_MODEL = ERNIE_DEFAULT_MODEL
    BACKEND_NAME = BackendType.Ernie


class AsyncErnieChatClient(AsyncOpenAICompatibleChatClient):
    DEFAULT_MODEL = ERNIE_DEFAULT_MODEL
    BACKEND_NAME = BackendType.Ernie
