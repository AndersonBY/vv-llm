# @Author: Bi Ying
# @Date:   2024-07-26 14:48:55
from ..types.enums import BackendType
from ..types.defaults import BAICHUAN_DEFAULT_MODEL
from .openai_compatible_client import OpenAICompatibleChatClient, AsyncOpenAICompatibleChatClient


class BaichuanChatClient(OpenAICompatibleChatClient):
    DEFAULT_MODEL = BAICHUAN_DEFAULT_MODEL
    BACKEND_NAME = BackendType.Baichuan


class AsyncBaichuanChatClient(AsyncOpenAICompatibleChatClient):
    DEFAULT_MODEL = BAICHUAN_DEFAULT_MODEL
    BACKEND_NAME = BackendType.Baichuan
