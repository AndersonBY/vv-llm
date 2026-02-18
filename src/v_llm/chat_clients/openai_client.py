# @Author: Bi Ying
# @Date:   2024-07-26 14:48:55
from ..types.enums import BackendType
from ..types.defaults import OPENAI_DEFAULT_MODEL
from .openai_compatible_client import OpenAICompatibleChatClient, AsyncOpenAICompatibleChatClient


class OpenAIChatClient(OpenAICompatibleChatClient):
    DEFAULT_MODEL = OPENAI_DEFAULT_MODEL
    BACKEND_NAME = BackendType.OpenAI


class AsyncOpenAIChatClient(AsyncOpenAICompatibleChatClient):
    DEFAULT_MODEL = OPENAI_DEFAULT_MODEL
    BACKEND_NAME = BackendType.OpenAI
