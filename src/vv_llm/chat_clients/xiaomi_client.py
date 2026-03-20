# @Author: Bi Ying
# @Date:   2024-07-26 14:48:55
from ..types.enums import BackendType
from ..types.defaults import XIAOMI_DEFAULT_MODEL
from .openai_compatible_client import OpenAICompatibleChatClient, AsyncOpenAICompatibleChatClient


class XiaomiChatClient(OpenAICompatibleChatClient):
    DEFAULT_MODEL = XIAOMI_DEFAULT_MODEL
    BACKEND_NAME = BackendType.Xiaomi


class AsyncXiaomiChatClient(AsyncOpenAICompatibleChatClient):
    DEFAULT_MODEL = XIAOMI_DEFAULT_MODEL
    BACKEND_NAME = BackendType.Xiaomi
