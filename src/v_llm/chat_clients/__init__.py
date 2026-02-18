# @Author: Bi Ying
# @Date:   2024-07-26 14:48:55
import httpx
from typing import overload, Literal, TYPE_CHECKING

from .base_client import BaseChatClient, BaseAsyncChatClient

from .yi_client import YiChatClient, AsyncYiChatClient
from .xai_client import XAIChatClient, AsyncXAIChatClient
from .groq_client import GroqChatClient, AsyncGroqChatClient
from .qwen_client import QwenChatClient, AsyncQwenChatClient
from .local_client import LocalChatClient, AsyncLocalChatClient
from .ernie_client import ErnieChatClient, AsyncErnieChatClient
from .gemini_client import GeminiChatClient, AsyncGeminiChatClient
from .openai_client import OpenAIChatClient, AsyncOpenAIChatClient
from .zhipuai_client import ZhiPuAIChatClient, AsyncZhiPuAIChatClient
from .minimax_client import MiniMaxChatClient, AsyncMiniMaxChatClient
from .mistral_client import MistralChatClient, AsyncMistralChatClient
from .stepfun_client import StepFunChatClient, AsyncStepFunChatClient
from .baichuan_client import BaichuanChatClient, AsyncBaichuanChatClient
from .moonshot_client import MoonshotChatClient, AsyncMoonshotChatClient
from .deepseek_client import DeepSeekChatClient, AsyncDeepSeekChatClient

from ..types import defaults as defs
from ..types.llm_parameters import NOT_GIVEN, NotGiven
from ..types.enums import BackendType, ContextLengthControlType
from .anthropic_client import AnthropicChatClient, AsyncAnthropicChatClient
from .utils import format_messages, get_token_counts, get_message_token_counts, ToolCallContentProcessor


if TYPE_CHECKING:
    from ..settings import Settings
    from ..types.settings import SettingsDict

SyncClientFactory = type[BaseChatClient]
AsyncClientFactory = type[BaseAsyncChatClient]

SYNC_BACKEND_MAP: dict[BackendType, SyncClientFactory] = {
    BackendType.Anthropic: AnthropicChatClient,
    BackendType.DeepSeek: DeepSeekChatClient,
    BackendType.Gemini: GeminiChatClient,
    BackendType.Groq: GroqChatClient,
    BackendType.Local: LocalChatClient,
    BackendType.MiniMax: MiniMaxChatClient,
    BackendType.Mistral: MistralChatClient,
    BackendType.Moonshot: MoonshotChatClient,
    BackendType.OpenAI: OpenAIChatClient,
    BackendType.Qwen: QwenChatClient,
    BackendType.Yi: YiChatClient,
    BackendType.ZhiPuAI: ZhiPuAIChatClient,
    BackendType.Baichuan: BaichuanChatClient,
    BackendType.StepFun: StepFunChatClient,
    BackendType.XAI: XAIChatClient,
    BackendType.Ernie: ErnieChatClient,
}

ASYNC_BACKEND_MAP: dict[BackendType, AsyncClientFactory] = {
    BackendType.Anthropic: AsyncAnthropicChatClient,
    BackendType.DeepSeek: AsyncDeepSeekChatClient,
    BackendType.Gemini: AsyncGeminiChatClient,
    BackendType.Groq: AsyncGroqChatClient,
    BackendType.Local: AsyncLocalChatClient,
    BackendType.MiniMax: AsyncMiniMaxChatClient,
    BackendType.Mistral: AsyncMistralChatClient,
    BackendType.Moonshot: AsyncMoonshotChatClient,
    BackendType.OpenAI: AsyncOpenAIChatClient,
    BackendType.Qwen: AsyncQwenChatClient,
    BackendType.Yi: AsyncYiChatClient,
    BackendType.ZhiPuAI: AsyncZhiPuAIChatClient,
    BackendType.Baichuan: AsyncBaichuanChatClient,
    BackendType.StepFun: AsyncStepFunChatClient,
    BackendType.XAI: AsyncXAIChatClient,
    BackendType.Ernie: AsyncErnieChatClient,
}


@overload
def create_chat_client(
    backend: Literal[BackendType.Anthropic],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.Client | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> AnthropicChatClient: ...


@overload
def create_chat_client(
    backend: Literal[BackendType.DeepSeek],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.Client | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> DeepSeekChatClient: ...


@overload
def create_chat_client(
    backend: Literal[BackendType.Gemini],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.Client | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> GeminiChatClient: ...


@overload
def create_chat_client(
    backend: Literal[BackendType.Groq],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.Client | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> GroqChatClient: ...


@overload
def create_chat_client(
    backend: Literal[BackendType.Local],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.Client | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> LocalChatClient: ...


@overload
def create_chat_client(
    backend: Literal[BackendType.MiniMax],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.Client | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> MiniMaxChatClient: ...


@overload
def create_chat_client(
    backend: Literal[BackendType.Mistral],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.Client | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> MistralChatClient: ...


@overload
def create_chat_client(
    backend: Literal[BackendType.Moonshot],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.Client | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> MoonshotChatClient: ...


@overload
def create_chat_client(
    backend: Literal[BackendType.OpenAI],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.Client | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> OpenAIChatClient: ...


@overload
def create_chat_client(
    backend: Literal[BackendType.Qwen],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.Client | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> QwenChatClient: ...


@overload
def create_chat_client(
    backend: Literal[BackendType.Yi],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.Client | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> YiChatClient: ...


@overload
def create_chat_client(
    backend: Literal[BackendType.ZhiPuAI],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.Client | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> ZhiPuAIChatClient: ...


@overload
def create_chat_client(
    backend: Literal[BackendType.Baichuan],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.Client | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> BaichuanChatClient: ...


@overload
def create_chat_client(
    backend: Literal[BackendType.StepFun],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.Client | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> StepFunChatClient: ...


@overload
def create_chat_client(
    backend: Literal[BackendType.XAI],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.Client | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> XAIChatClient: ...


@overload
def create_chat_client(
    backend: Literal[BackendType.Ernie],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.Client | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> ErnieChatClient: ...


@overload
def create_chat_client(
    backend: BackendType,
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.Client | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> BaseChatClient: ...


def create_chat_client(
    backend: BackendType,
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.Client | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> BaseChatClient:
    if backend not in SYNC_BACKEND_MAP:
        raise ValueError(f"Unsupported backend: {backend}")

    ClientClass = SYNC_BACKEND_MAP[backend]
    if model is None:
        model = ClientClass.DEFAULT_MODEL
    return ClientClass(
        model=model,
        stream=stream,
        temperature=temperature,
        context_length_control=context_length_control,
        random_endpoint=random_endpoint,
        endpoint_id=endpoint_id,
        http_client=http_client,
        settings=settings,
        **kwargs,
    )


@overload
def create_async_chat_client(
    backend: Literal[BackendType.Anthropic],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.AsyncClient | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> AsyncAnthropicChatClient: ...


@overload
def create_async_chat_client(
    backend: Literal[BackendType.DeepSeek],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.AsyncClient | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> AsyncDeepSeekChatClient: ...


@overload
def create_async_chat_client(
    backend: Literal[BackendType.Gemini],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.AsyncClient | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> AsyncGeminiChatClient: ...


@overload
def create_async_chat_client(
    backend: Literal[BackendType.Groq],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.AsyncClient | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> AsyncGroqChatClient: ...


@overload
def create_async_chat_client(
    backend: Literal[BackendType.Local],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.AsyncClient | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> AsyncLocalChatClient: ...


@overload
def create_async_chat_client(
    backend: Literal[BackendType.MiniMax],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.AsyncClient | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> AsyncMiniMaxChatClient: ...


@overload
def create_async_chat_client(
    backend: Literal[BackendType.Mistral],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.AsyncClient | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> AsyncMistralChatClient: ...


@overload
def create_async_chat_client(
    backend: Literal[BackendType.Moonshot],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.AsyncClient | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> AsyncMoonshotChatClient: ...


@overload
def create_async_chat_client(
    backend: Literal[BackendType.OpenAI],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.AsyncClient | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> AsyncOpenAIChatClient: ...


@overload
def create_async_chat_client(
    backend: Literal[BackendType.Qwen],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.AsyncClient | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> AsyncQwenChatClient: ...


@overload
def create_async_chat_client(
    backend: Literal[BackendType.Yi],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.AsyncClient | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> AsyncYiChatClient: ...


@overload
def create_async_chat_client(
    backend: Literal[BackendType.ZhiPuAI],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.AsyncClient | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> AsyncZhiPuAIChatClient: ...


@overload
def create_async_chat_client(
    backend: Literal[BackendType.Baichuan],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.AsyncClient | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> AsyncBaichuanChatClient: ...


@overload
def create_async_chat_client(
    backend: Literal[BackendType.StepFun],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.AsyncClient | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> AsyncStepFunChatClient: ...


@overload
def create_async_chat_client(
    backend: Literal[BackendType.XAI],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.AsyncClient | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> AsyncXAIChatClient: ...


@overload
def create_async_chat_client(
    backend: Literal[BackendType.Ernie],
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.AsyncClient | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> AsyncErnieChatClient: ...


@overload
def create_async_chat_client(
    backend: BackendType,
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.AsyncClient | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> BaseAsyncChatClient: ...


def create_async_chat_client(
    backend: BackendType,
    model: str | None = None,
    stream: bool = False,
    temperature: float | None | NotGiven = NOT_GIVEN,
    context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
    random_endpoint: bool = True,
    endpoint_id: str = "",
    http_client: httpx.AsyncClient | None = None,
    settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    **kwargs,
) -> BaseAsyncChatClient:
    if backend not in ASYNC_BACKEND_MAP:
        raise ValueError(f"Unsupported backend: {backend}")

    ClientClass = ASYNC_BACKEND_MAP[backend]
    if model is None:
        model = ClientClass.DEFAULT_MODEL
    return ClientClass(
        model=model,
        stream=stream,
        temperature=temperature,
        context_length_control=context_length_control,
        random_endpoint=random_endpoint,
        endpoint_id=endpoint_id,
        http_client=http_client,
        settings=settings,
        **kwargs,
    )


__all__ = [
    "BackendType",
    "format_messages",
    "get_token_counts",
    "create_chat_client",
    "create_async_chat_client",
    "get_message_token_counts",
    "ToolCallContentProcessor",
]
