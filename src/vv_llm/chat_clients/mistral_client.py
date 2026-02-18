# @Author: Bi Ying
# @Date:   2024-07-26 14:48:55
from ..types.enums import BackendType
from ..types.defaults import MISTRAL_DEFAULT_MODEL
from .openai_compatible_client import OpenAICompatibleChatClient, AsyncOpenAICompatibleChatClient


class MistralChatClient(OpenAICompatibleChatClient):
    DEFAULT_MODEL = MISTRAL_DEFAULT_MODEL
    BACKEND_NAME = BackendType.Mistral


class AsyncMistralChatClient(AsyncOpenAICompatibleChatClient):
    DEFAULT_MODEL = MISTRAL_DEFAULT_MODEL
    BACKEND_NAME = BackendType.Mistral
