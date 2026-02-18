# @Author: Bi Ying
# @Date:   2024-07-26 14:48:55
from ..types.enums import BackendType
from ..types.defaults import GROQ_DEFAULT_MODEL
from .openai_compatible_client import OpenAICompatibleChatClient, AsyncOpenAICompatibleChatClient


class GroqChatClient(OpenAICompatibleChatClient):
    DEFAULT_MODEL = GROQ_DEFAULT_MODEL
    BACKEND_NAME = BackendType.Groq


class AsyncGroqChatClient(AsyncOpenAICompatibleChatClient):
    DEFAULT_MODEL = GROQ_DEFAULT_MODEL
    BACKEND_NAME = BackendType.Groq
