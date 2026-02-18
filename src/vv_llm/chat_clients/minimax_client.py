from ..types.enums import BackendType
from ..types.defaults import MINIMAX_DEFAULT_MODEL
from .openai_compatible_client import OpenAICompatibleChatClient, AsyncOpenAICompatibleChatClient


class MiniMaxChatClient(OpenAICompatibleChatClient):
    DEFAULT_MODEL = MINIMAX_DEFAULT_MODEL
    BACKEND_NAME = BackendType.MiniMax


class AsyncMiniMaxChatClient(AsyncOpenAICompatibleChatClient):
    DEFAULT_MODEL = MINIMAX_DEFAULT_MODEL
    BACKEND_NAME = BackendType.MiniMax
