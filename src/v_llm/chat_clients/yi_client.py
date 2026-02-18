# @Author: Bi Ying
# @Date:   2024-07-26 14:48:55
from ..types.enums import BackendType
from ..types.defaults import YI_DEFAULT_MODEL
from .openai_compatible_client import OpenAICompatibleChatClient, AsyncOpenAICompatibleChatClient


class YiChatClient(OpenAICompatibleChatClient):
    DEFAULT_MODEL = YI_DEFAULT_MODEL
    BACKEND_NAME = BackendType.Yi


class AsyncYiChatClient(AsyncOpenAICompatibleChatClient):
    DEFAULT_MODEL = YI_DEFAULT_MODEL
    BACKEND_NAME = BackendType.Yi
