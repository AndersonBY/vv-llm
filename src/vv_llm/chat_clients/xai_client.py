# @Author: Bi Ying
# @Date:   2024-07-26 14:48:55
from ..types.enums import BackendType
from ..types.defaults import XAI_DEFAULT_MODEL
from .openai_compatible_client import OpenAICompatibleChatClient, AsyncOpenAICompatibleChatClient


class XAIChatClient(OpenAICompatibleChatClient):
    DEFAULT_MODEL = XAI_DEFAULT_MODEL
    BACKEND_NAME = BackendType.XAI


class AsyncXAIChatClient(AsyncOpenAICompatibleChatClient):
    DEFAULT_MODEL = XAI_DEFAULT_MODEL
    BACKEND_NAME = BackendType.XAI
