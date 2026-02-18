from ..types.enums import BackendType
from ..types.defaults import GEMINI_DEFAULT_MODEL
from .openai_compatible_client import OpenAICompatibleChatClient, AsyncOpenAICompatibleChatClient


class GeminiChatClient(OpenAICompatibleChatClient):
    DEFAULT_MODEL = GEMINI_DEFAULT_MODEL
    BACKEND_NAME = BackendType.Gemini


class AsyncGeminiChatClient(AsyncOpenAICompatibleChatClient):
    DEFAULT_MODEL = GEMINI_DEFAULT_MODEL
    BACKEND_NAME = BackendType.Gemini
