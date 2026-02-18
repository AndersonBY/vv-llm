# @Author: Bi Ying
# @Date:   2024-07-26 14:48:55
from ..types.enums import BackendType
from ..types.defaults import MOONSHOT_DEFAULT_MODEL
from .openai_compatible_client import OpenAICompatibleChatClient, AsyncOpenAICompatibleChatClient


class MoonshotChatClient(OpenAICompatibleChatClient):
    DEFAULT_MODEL = MOONSHOT_DEFAULT_MODEL
    BACKEND_NAME = BackendType.Moonshot


class AsyncMoonshotChatClient(AsyncOpenAICompatibleChatClient):
    DEFAULT_MODEL = MOONSHOT_DEFAULT_MODEL
    BACKEND_NAME = BackendType.Moonshot
