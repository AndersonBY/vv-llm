# @Author: Bi Ying
# @Date:   2024-07-26 14:48:55
from ..types.enums import BackendType
from ..types.defaults import STEPFUN_DEFAULT_MODEL
from .openai_compatible_client import OpenAICompatibleChatClient, AsyncOpenAICompatibleChatClient


class StepFunChatClient(OpenAICompatibleChatClient):
    DEFAULT_MODEL = STEPFUN_DEFAULT_MODEL
    BACKEND_NAME = BackendType.StepFun


class AsyncStepFunChatClient(AsyncOpenAICompatibleChatClient):
    DEFAULT_MODEL = STEPFUN_DEFAULT_MODEL
    BACKEND_NAME = BackendType.StepFun
