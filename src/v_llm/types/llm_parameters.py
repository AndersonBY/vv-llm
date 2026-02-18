# @Author: Bi Ying
# @Date:   2024-07-26 23:48:04
from collections.abc import Iterable
from typing import Literal
from typing_extensions import TypedDict, NotRequired

import httpx
from pydantic import BaseModel, Field

from anthropic._types import NotGiven as AnthropicNotGiven
from anthropic._types import NOT_GIVEN as ANTHROPIC_NOT_GIVEN
from anthropic.types import ToolParam as AnthropicToolParam
from anthropic.types import ThinkingConfigParam, ThinkingConfigEnabledParam
from anthropic.types.message_create_params import ToolChoice as AnthropicToolChoice

from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from openai._types import NotGiven as OpenAINotGiven
from openai._types import NOT_GIVEN as OPENAI_NOT_GIVEN
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_function_tool_param import ChatCompletionFunctionToolParam
from openai.types.chat.chat_completion_stream_options_param import ChatCompletionStreamOptionsParam
from openai.types.chat.chat_completion_tool_choice_option_param import ChatCompletionToolChoiceOptionParam
from openai.types.completion_usage import CompletionTokensDetails, PromptTokensDetails

from . import defaults as defs
from .settings import EndpointOptionDict


class VectorVeinTextMessage(TypedDict):
    mid: NotRequired[str]
    author_type: Literal["U", "A"]  # "U" for user, "A" for assistant
    content_type: Literal["TXT"]
    status: NotRequired[str]
    create_time: NotRequired[int]
    update_time: NotRequired[int]
    metadata: NotRequired[dict]
    content: dict[str, str]
    attachments: NotRequired[list[str]]


class VectorVeinWorkflowMessage(TypedDict):
    mid: NotRequired[str]
    author_type: Literal["U", "A"]  # "U" for user, "A" for assistant
    content_type: Literal["WKF"]
    status: str
    create_time: NotRequired[int]
    update_time: NotRequired[int]
    metadata: dict
    content: dict[str, str]
    attachments: NotRequired[list[str]]


VectorVeinMessage = VectorVeinTextMessage | VectorVeinWorkflowMessage


class EndpointSetting(BaseModel):
    id: str = Field(..., description="The id of the endpoint.")
    enabled: bool = Field(True, description="Whether the endpoint is enabled.")
    region: str | None = Field(None, description="The region for the endpoint.")
    api_base: str | None = Field(None, description="The base URL for the API.")
    api_key: str | None = Field(None, description="The API key for authentication.")
    response_api: bool = Field(False, description="Whether to use the new Responses API style.")
    endpoint_type: (
        Literal[
            "default",
            "openai",
            "openai_azure",
            "openai_vertex",
            "anthropic",
            "anthropic_vertex",
            "anthropic_bedrock",
        ]
        | None
    ) = Field(
        "default",
        description="The type of endpoint. Set to 'default' will determine the type automatically.",
    )
    credentials: dict | None = Field(None, description="Additional credentials if needed.")
    is_azure: bool = Field(False, description="Indicates if the endpoint is for Azure.")
    is_vertex: bool = Field(False, description="Indicates if the endpoint is for Vertex.")
    is_bedrock: bool = Field(False, description="Indicates if the endpoint is for Bedrock.")
    rpm: int = Field(description="Requests per minute.", default=defs.ENDPOINT_RPM)
    tpm: int = Field(description="Tokens per minute.", default=defs.ENDPOINT_TPM)
    concurrent_requests: int = Field(
        description="Whether to use concurrent requests for the LLM service.",
        default=defs.ENDPOINT_CONCURRENT_REQUESTS,
    )
    proxy: str | None = Field(None, description="The proxy URL for the endpoint.")
    access_token: str | None = Field(None, description="Cached GCP access token for Vertex endpoints.")
    access_token_expires_at: float | None = Field(None, description="Expiry timestamp (Unix time) of the cached access token.")

    def model_list(self):
        http_client = httpx.Client(proxy=self.proxy) if self.proxy is not None else None

        if self.is_azure:
            if self.api_base is None:
                raise ValueError("Azure endpoint is not set")
            _client = AzureOpenAI(
                azure_endpoint=self.api_base,
                api_key=self.api_key,
                api_version="2025-01-01-preview",
                http_client=http_client,
            )
        else:
            _client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                http_client=http_client,
            )

        return _client.models.list().model_dump()

    async def amodel_list(self):
        http_client = httpx.AsyncClient(proxy=self.proxy) if self.proxy is not None else None

        if self.is_azure:
            if self.api_base is None:
                raise ValueError("Azure endpoint is not set")
            _client = AsyncAzureOpenAI(
                azure_endpoint=self.api_base,
                api_key=self.api_key,
                api_version="2025-01-01-preview",
                http_client=http_client,
            )
        else:
            _client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                http_client=http_client,
            )

        return (await _client.models.list()).model_dump()


class ModelSetting(BaseModel):
    id: str = Field(..., description="The id of the model.")
    enabled: bool = Field(True, description="Whether the model is enabled.")
    endpoints: list[str | EndpointOptionDict] = Field(default_factory=list, description="Available endpoints for the model.")
    function_call_available: bool = Field(False, description="Indicates if function call is available.")
    response_format_available: bool = Field(False, description="Indicates if response format is available.")
    native_multimodal: bool = Field(False, description="Indicates if the model is a native multimodal model.")
    context_length: int = Field(32768, description="The context length for the model.")
    max_output_tokens: int | None = Field(None, description="Maximum number of output tokens allowed.")


class BackendSettings(BaseModel):
    models: dict[str, ModelSetting] = Field(default_factory=dict)
    default_endpoint: str | None = Field(default_factory=lambda: None, description="The default endpoint for the model.")

    def get_model_setting(self, model_name: str) -> ModelSetting:
        if model_name in self.models:
            model_setting = self.models[model_name]
            if len(model_setting.endpoints) == 0 and self.default_endpoint is not None:
                model_setting.endpoints = [self.default_endpoint]
            return model_setting
        else:
            raise ValueError(f"Model {model_name} not found in {self.models}")

    def update_models(self, default_models: dict[str, dict], input_models: dict[str, dict]):
        updated_models: dict[str, ModelSetting] = {}
        for model_name, model_data in default_models.items():
            updated_model = ModelSetting(**model_data)
            if model_name in input_models:
                updated_model = updated_model.model_copy(update=input_models[model_name])
            updated_models[model_name] = updated_model

        # Add any new models from input that weren't in defaults
        for model_name, model_data in input_models.items():
            if model_name not in updated_models:
                updated_models[model_name] = ModelSetting(**model_data)

        self.models = updated_models


class Usage(BaseModel):
    completion_tokens: int

    prompt_tokens: int

    total_tokens: int

    completion_tokens_details: CompletionTokensDetails | None = None
    """Breakdown of tokens used in a completion."""

    prompt_tokens_details: PromptTokensDetails | None = None
    """Breakdown of tokens used in the prompt."""


class ChatCompletionMessage(BaseModel):
    content: str | None = None

    reasoning_content: str | None = None

    raw_content: list[dict] | None = None

    tool_calls: list[ChatCompletionMessageToolCall] | None = None
    """The tool calls generated by the model, such as function calls."""

    function_call_arguments: dict | None = None

    usage: Usage | None = None


class ChatCompletionDeltaMessage(BaseModel):
    content: str | None = None

    reasoning_content: str | None = None

    raw_content: dict | None = None

    tool_calls: list[ChoiceDeltaToolCall] | None = None
    """The tool calls generated by the model, such as function calls."""

    function_call_arguments: dict | None = None

    usage: Usage | None = None


NotGiven = AnthropicNotGiven | OpenAINotGiven

NOT_GIVEN = OPENAI_NOT_GIVEN

OpenAIToolParam = ChatCompletionFunctionToolParam
ToolParam = OpenAIToolParam

Tools = Iterable[ToolParam]

ToolChoice = ChatCompletionToolChoiceOptionParam


__all__ = [
    "EndpointSetting",
    "ModelSetting",
    "BackendSettings",
    "Usage",
    "ChatCompletionMessage",
    "ChatCompletionMessageParam",
    "ChatCompletionDeltaMessage",
    "ChatCompletionStreamOptionsParam",
    "NotGiven",
    "NOT_GIVEN",
    "OpenAIToolParam",
    "ToolParam",
    "Tools",
    "ToolChoice",
    "AnthropicToolParam",
    "AnthropicToolChoice",
    "OPENAI_NOT_GIVEN",
    "ANTHROPIC_NOT_GIVEN",
    "ResponseFormat",
    "ThinkingConfigParam",
    "ThinkingConfigEnabledParam",
    "VectorVeinMessage",
    "VectorVeinTextMessage",
    "VectorVeinWorkflowMessage",
]
