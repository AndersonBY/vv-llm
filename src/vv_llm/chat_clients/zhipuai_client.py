# @Author: Bi Ying
# @Date:   2024-07-26 14:48:55
from ..types.enums import BackendType
from ..types.defaults import ZHIPUAI_DEFAULT_MODEL
from .openai_compatible_client import OpenAICompatibleChatClient, AsyncOpenAICompatibleChatClient


class ZhiPuAIChatClient(OpenAICompatibleChatClient):
    DEFAULT_MODEL = ZHIPUAI_DEFAULT_MODEL
    BACKEND_NAME = BackendType.ZhiPuAI


class AsyncZhiPuAIChatClient(AsyncOpenAICompatibleChatClient):
    DEFAULT_MODEL = ZHIPUAI_DEFAULT_MODEL
    BACKEND_NAME = BackendType.ZhiPuAI
