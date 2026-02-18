# @Author: Bi Ying
# @Date:   2024-07-26 14:48:55
from collections.abc import Generator, AsyncGenerator, Iterable
from typing import Any, TYPE_CHECKING, overload, Literal, cast

import httpx
from openai._types import NotGiven as OpenAINotGiven
from openai._types import NOT_GIVEN as OPENAI_NOT_GIVEN
from openai._types import Headers, Query, Body
from openai.types.shared_params.metadata import Metadata
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.chat.chat_completion_modality import ChatCompletionModality
from openai.types.chat.chat_completion_audio_param import ChatCompletionAudioParam
from openai.types.chat.chat_completion_reasoning_effort import ChatCompletionReasoningEffort
from openai.types.chat.chat_completion_stream_options_param import ChatCompletionStreamOptionsParam
from openai.types.chat.chat_completion_prediction_content_param import ChatCompletionPredictionContentParam
from anthropic import (
    Anthropic,
    AnthropicVertex,
    AsyncAnthropic,
    AsyncAnthropicVertex,
    AnthropicBedrock,
    AsyncAnthropicBedrock,
)
from anthropic._types import NOT_GIVEN, omit, Omit
from anthropic._exceptions import APIStatusError as AnthropicAPIStatusError
from anthropic._exceptions import APIConnectionError as AnthropicAPIConnectionError
from anthropic.types import (
    ThinkingConfigParam,
)

from ..types import defaults as defs
from .utils import cutoff_messages, get_message_token_counts
from .base_client import BaseChatClient, BaseAsyncChatClient
from .message_normalizer import format_messages_alternate, refactor_into_openai_messages
from .openai_compatible_client import OpenAICompatibleChatClient, AsyncOpenAICompatibleChatClient
from .stream_event_adapter import (
    adapt_anthropic_stream_event,
    build_anthropic_completion_message,
    init_anthropic_stream_state,
)
from .tool_call_parser import refactor_tool_choice, refactor_tool_use_params
from ..types.exception import APIStatusError, APIConnectionError
from ..types.enums import ContextLengthControlType, BackendType
from ..utilities.gcp_token import get_token_with_cache
from ..types.llm_parameters import NotGiven, ToolParam, ToolChoice, AnthropicToolParam, ChatCompletionMessage, ChatCompletionDeltaMessage

if TYPE_CHECKING:
    from ..settings import Settings
    from ..types.settings import SettingsDict


class AnthropicChatClient(BaseChatClient):
    DEFAULT_MODEL: str = defs.ANTHROPIC_DEFAULT_MODEL
    BACKEND_NAME: BackendType = BackendType.Anthropic

    def __init__(
        self,
        model: str = defs.ANTHROPIC_DEFAULT_MODEL,
        stream: bool = True,
        temperature: float | None | NotGiven = NOT_GIVEN,
        context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
        random_endpoint: bool = True,
        endpoint_id: str = "",
        http_client: httpx.Client | None = None,
        backend_name: str | None = None,
        settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    ):
        super().__init__(
            model,
            stream,
            temperature,
            context_length_control,
            random_endpoint,
            endpoint_id,
            http_client,
            backend_name,
            settings,
        )
        self.model_id = None
        self.endpoint = None
        self._cached_raw_client = None

    @property
    def raw_client(self):
        self.endpoint, self.model_id = self._set_endpoint()

        if self.endpoint.proxy and self.http_client is None:
            self.http_client = httpx.Client(proxy=self.endpoint.proxy)

        if self.endpoint.is_vertex or self.endpoint.endpoint_type == "anthropic_vertex":
            # Vertex needs token refresh on each access
            if self.endpoint.credentials is None:
                raise ValueError("Anthropic Vertex endpoint requires credentials")

            access_token, expires_at = get_token_with_cache(
                credentials=self.endpoint.credentials,
                proxy=self.endpoint.proxy,
                cached_token=self.endpoint.access_token,
                cached_expires_at=self.endpoint.access_token_expires_at,
            )
            self.endpoint.access_token = access_token
            self.endpoint.access_token_expires_at = expires_at

            if self.endpoint.api_base is None:
                base_url = None
            else:
                base_url = f"{self.endpoint.api_base}{self.endpoint.region}-aiplatform/v1"

            region = NOT_GIVEN if self.endpoint.region is None else self.endpoint.region
            return AnthropicVertex(
                region=region,
                base_url=base_url,
                project_id=self.endpoint.credentials.get("quota_project_id", NOT_GIVEN),
                access_token=access_token,
                http_client=self.http_client,
            )
        elif self.endpoint.is_bedrock or self.endpoint.endpoint_type == "anthropic_bedrock":
            if self.endpoint.credentials is None:
                raise ValueError("Anthropic Bedrock endpoint requires credentials")
            return AnthropicBedrock(
                aws_access_key=self.endpoint.credentials.get("access_key"),
                aws_secret_key=self.endpoint.credentials.get("secret_key"),
                aws_region=self.endpoint.region,
                base_url=self.endpoint.api_base,
                http_client=self.http_client,
            )
        elif self.endpoint.endpoint_type in ("default", "anthropic"):
            if self._cached_raw_client is None:
                self._cached_raw_client = Anthropic(
                    api_key=self.endpoint.api_key,
                    base_url=self.endpoint.api_base,
                    http_client=self.http_client,
                )
            return self._cached_raw_client
        else:
            if self._cached_raw_client is None:
                self._cached_raw_client = OpenAICompatibleChatClient(
                    model=self.model,
                    stream=self.stream,
                    temperature=cast(Any, self.temperature),
                    context_length_control=self.context_length_control,
                    random_endpoint=self.random_endpoint,
                    endpoint_id=self.endpoint_id,
                    http_client=self.http_client,
                    backend_name=self.BACKEND_NAME,
                ).raw_client
            return self._cached_raw_client

    @overload
    def create_completion(
        self,
        *,
        messages: list,
        model: str | None = None,
        stream: Literal[False] = False,
        temperature: float | None | NotGiven = OPENAI_NOT_GIVEN,
        max_tokens: int | None | NotGiven = None,
        tools: Iterable[ToolParam] | NotGiven = OPENAI_NOT_GIVEN,
        tool_choice: ToolChoice | NotGiven = OPENAI_NOT_GIVEN,
        response_format: ResponseFormat | NotGiven = OPENAI_NOT_GIVEN,
        stream_options: ChatCompletionStreamOptionsParam | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven | None = OPENAI_NOT_GIVEN,
        skip_cutoff: bool = False,
        audio: ChatCompletionAudioParam | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        frequency_penalty: float | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        logit_bias: dict[str, int] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        logprobs: bool | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        max_completion_tokens: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        metadata: Metadata | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        modalities: list[ChatCompletionModality] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        n: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        parallel_tool_calls: bool | OpenAINotGiven = OPENAI_NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        presence_penalty: float | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        thinking: ThinkingConfigParam | None | NotGiven = OPENAI_NOT_GIVEN,
        seed: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        service_tier: Literal["auto", "default"] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        stop: str | list[str] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        store: bool | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        top_logprobs: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        user: str | OpenAINotGiven = OPENAI_NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | OpenAINotGiven = OPENAI_NOT_GIVEN,
    ) -> ChatCompletionMessage:
        pass

    @overload
    def create_completion(
        self,
        *,
        messages: list,
        model: str | None = None,
        stream: Literal[True],
        temperature: float | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = None,
        tools: Iterable[ToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
        response_format: ResponseFormat | NotGiven = NOT_GIVEN,
        stream_options: ChatCompletionStreamOptionsParam | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        skip_cutoff: bool = False,
        audio: ChatCompletionAudioParam | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        frequency_penalty: float | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        logit_bias: dict[str, int] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        logprobs: bool | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        max_completion_tokens: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        metadata: Metadata | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        modalities: list[ChatCompletionModality] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        n: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        parallel_tool_calls: bool | OpenAINotGiven = OPENAI_NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        presence_penalty: float | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        thinking: ThinkingConfigParam | None | NotGiven = NOT_GIVEN,
        seed: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        service_tier: Literal["auto", "default"] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        stop: str | list[str] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        store: bool | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        top_logprobs: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        user: str | OpenAINotGiven = OPENAI_NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | OpenAINotGiven = OPENAI_NOT_GIVEN,
    ) -> Generator[ChatCompletionDeltaMessage, None, None]:
        pass

    @overload
    def create_completion(
        self,
        *,
        messages: list,
        model: str | None = None,
        stream: bool,
        temperature: float | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = None,
        tools: Iterable[ToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
        response_format: ResponseFormat | NotGiven = NOT_GIVEN,
        stream_options: ChatCompletionStreamOptionsParam | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        skip_cutoff: bool = False,
        audio: ChatCompletionAudioParam | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        frequency_penalty: float | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        logit_bias: dict[str, int] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        logprobs: bool | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        max_completion_tokens: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        metadata: Metadata | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        modalities: list[ChatCompletionModality] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        n: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        parallel_tool_calls: bool | OpenAINotGiven = OPENAI_NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        presence_penalty: float | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        thinking: ThinkingConfigParam | None | NotGiven = NOT_GIVEN,
        seed: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        service_tier: Literal["auto", "default"] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        stop: str | list[str] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        store: bool | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        top_logprobs: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        user: str | OpenAINotGiven = OPENAI_NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | OpenAINotGiven = OPENAI_NOT_GIVEN,
    ) -> ChatCompletionMessage | Generator[ChatCompletionDeltaMessage, Any, None]:
        pass

    def create_completion(
        self,
        *,
        messages: list,
        model: str | None = None,
        stream: Literal[False] | Literal[True] = False,
        temperature: float | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = None,
        tools: Iterable[ToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
        response_format: ResponseFormat | NotGiven = NOT_GIVEN,
        stream_options: ChatCompletionStreamOptionsParam | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        skip_cutoff: bool = False,
        audio: ChatCompletionAudioParam | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        frequency_penalty: float | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        logit_bias: dict[str, int] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        logprobs: bool | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        max_completion_tokens: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        metadata: Metadata | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        modalities: list[ChatCompletionModality] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        n: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        parallel_tool_calls: bool | OpenAINotGiven = OPENAI_NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        presence_penalty: float | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        thinking: ThinkingConfigParam | None | NotGiven = NOT_GIVEN,
        seed: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        service_tier: Literal["auto", "default"] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        stop: str | list[str] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        store: bool | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        top_logprobs: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        user: str | OpenAINotGiven = OPENAI_NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | OpenAINotGiven = OPENAI_NOT_GIVEN,
    ):
        if model is not None:
            self.model = model
        if stream is not None:
            self.stream = stream
        if temperature is not None:
            self.temperature = temperature

        self.model_setting = self.backend_settings.models[self.model]
        if self.model_id is None:
            self.model_id = self.model_setting.id

        self.endpoint, self.model_id = self._set_endpoint()

        if self.endpoint.endpoint_type and self.endpoint.endpoint_type.startswith("openai"):
            _tools = OPENAI_NOT_GIVEN if tools is NOT_GIVEN else tools
            _tool_choice = OPENAI_NOT_GIVEN if tool_choice is NOT_GIVEN else tool_choice

            formatted_messages = refactor_into_openai_messages(messages)

            if self.stream:

                def _generator():
                    openai_client = cast(
                        Any,
                        OpenAICompatibleChatClient(
                            model=self.model,
                            stream=True,
                            temperature=cast(Any, self.temperature),
                            context_length_control=self.context_length_control,
                            random_endpoint=self.random_endpoint,
                            endpoint_id=self.endpoint_id,
                            http_client=self.http_client,
                            backend_name=self.BACKEND_NAME,
                        ),
                    )
                    response = openai_client.create_completion(
                        messages=formatted_messages,
                        model=model,
                        stream=True,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=_tools,
                        tool_choice=_tool_choice,
                        response_format=response_format,
                        stream_options=stream_options,
                        top_p=top_p,
                        skip_cutoff=skip_cutoff,
                        audio=audio,
                        frequency_penalty=frequency_penalty,
                        logit_bias=logit_bias,
                        logprobs=logprobs,
                        max_completion_tokens=max_completion_tokens,
                        metadata=metadata,
                        modalities=modalities,
                        n=n,
                        parallel_tool_calls=parallel_tool_calls,
                        prediction=prediction,
                        presence_penalty=presence_penalty,
                        reasoning_effort=reasoning_effort,
                        seed=seed,
                        service_tier=service_tier,
                        stop=stop,
                        store=store,
                        top_logprobs=top_logprobs,
                        user=user,
                        extra_headers=extra_headers,
                        extra_query=extra_query,
                        extra_body=extra_body,
                        timeout=timeout,
                    )
                    yield from response

                return _generator()
            else:
                openai_client = cast(
                    Any,
                    OpenAICompatibleChatClient(
                        model=self.model,
                        stream=False,
                        temperature=cast(Any, self.temperature),
                        context_length_control=self.context_length_control,
                        random_endpoint=self.random_endpoint,
                        endpoint_id=self.endpoint_id,
                        http_client=self.http_client,
                        backend_name=self.BACKEND_NAME,
                    ),
                )
                return openai_client.create_completion(
                    messages=formatted_messages,
                    model=model,
                    stream=False,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=_tools,
                    tool_choice=_tool_choice,
                    response_format=response_format,
                    top_p=top_p,
                    skip_cutoff=skip_cutoff,
                    audio=audio,
                    frequency_penalty=frequency_penalty,
                    logit_bias=logit_bias,
                    logprobs=logprobs,
                    max_completion_tokens=max_completion_tokens,
                    metadata=metadata,
                    modalities=modalities,
                    n=n,
                    parallel_tool_calls=parallel_tool_calls,
                    prediction=prediction,
                    presence_penalty=presence_penalty,
                    reasoning_effort=reasoning_effort,
                    seed=seed,
                    service_tier=service_tier,
                    stop=stop,
                    store=store,
                    top_logprobs=top_logprobs,
                    user=user,
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    extra_body=extra_body,
                    timeout=timeout,
                )

        raw_client = self.raw_client  # 调用完 self.raw_client 后，self.model_id 会被赋值
        assert isinstance(raw_client, Anthropic | AnthropicVertex | AnthropicBedrock)

        _tools = omit if isinstance(tools, NotGiven) else tools
        _tool_choice = omit if isinstance(tool_choice, NotGiven) else tool_choice
        _top_p = omit if isinstance(top_p, NotGiven) or top_p is None else top_p
        if isinstance(self.temperature, NotGiven) or self.temperature is None:
            self.temperature = omit
        _thinking = omit if isinstance(thinking, NotGiven) or thinking is None else thinking

        if messages[0].get("role") == "system":
            system_content = messages[0]["content"]
            # Preserve list format for system prompt to support cache_control
            if isinstance(system_content, list):
                system_prompt = system_content  # Keep as list to preserve cache_control
            else:
                system_prompt = system_content  # String format
            messages = messages[1:]
        else:
            system_prompt = ""

        if not skip_cutoff and self.context_length_control == ContextLengthControlType.Latest:
            messages = cutoff_messages(
                messages,
                max_count=self.model_setting.context_length,
                backend=self.BACKEND_NAME,
                model=self.model,
            )

        messages = format_messages_alternate(messages)

        tools_params: list[AnthropicToolParam] | Omit = refactor_tool_use_params(_tools) if _tools else omit
        tool_choice_param = omit
        if _tool_choice:
            tool_choice_param = refactor_tool_choice(_tool_choice)

        if not max_tokens:
            max_output_tokens = self.model_setting.max_output_tokens
            native_multimodal = self.model_setting.native_multimodal
            token_counts = get_message_token_counts(messages=messages, tools=tools, model=self.model, native_multimodal=native_multimodal)
            if max_output_tokens is not None:
                max_tokens = self.model_setting.context_length - token_counts
                max_tokens = min(max(max_tokens, 1), max_output_tokens)
            else:
                max_tokens = self.model_setting.context_length - token_counts

        self._acquire_rate_limit(self.endpoint, self.model, messages)

        if self.stream:
            try:
                stream_response = raw_client.messages.create(
                    max_tokens=max_tokens,
                    messages=messages,
                    model=self.model_id,
                    stream=True,
                    system=system_prompt,
                    temperature=self.temperature,
                    tools=tools_params,
                    tool_choice=tool_choice_param,
                    top_p=_top_p,
                    thinking=_thinking,
                )
            except AnthropicAPIStatusError as e:
                raise APIStatusError(message=e.message, response=e.response, body=e.body) from e
            except AnthropicAPIConnectionError as e:
                raise APIConnectionError(message=e.message, request=e.request) from e

            def generator():
                stream_state = init_anthropic_stream_state()
                for chunk in stream_response:
                    message = adapt_anthropic_stream_event(chunk, stream_state)
                    if message is not None:
                        yield message

            return generator()
        else:
            try:
                response = raw_client.messages.create(
                    model=self.model_id,
                    messages=messages,
                    system=system_prompt,
                    stream=False,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    tools=tools_params,
                    tool_choice=tool_choice_param,
                    top_p=_top_p,
                    thinking=_thinking,
                )
            except AnthropicAPIStatusError as e:
                raise APIStatusError(message=e.message, response=e.response, body=e.body) from e
            except AnthropicAPIConnectionError as e:
                raise APIConnectionError(message=e.message, request=e.request) from e

            return build_anthropic_completion_message(list(response.content), response.usage)


class AsyncAnthropicChatClient(BaseAsyncChatClient):
    DEFAULT_MODEL: str = defs.ANTHROPIC_DEFAULT_MODEL
    BACKEND_NAME: BackendType = BackendType.Anthropic

    def __init__(
        self,
        model: str = defs.ANTHROPIC_DEFAULT_MODEL,
        stream: bool = True,
        temperature: float | None | NotGiven = NOT_GIVEN,
        context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
        random_endpoint: bool = True,
        endpoint_id: str = "",
        http_client: httpx.AsyncClient | None = None,
        backend_name: str | None = None,
        settings: "Settings | SettingsDict | None" = None,  # Use default settings if not provided
    ):
        super().__init__(
            model,
            stream,
            temperature,
            context_length_control,
            random_endpoint,
            endpoint_id,
            http_client,
            backend_name,
            settings,
        )
        self.model_id = None
        self.endpoint = None
        self._cached_raw_client = None

    @property
    def raw_client(self):
        self.endpoint, self.model_id = self._set_endpoint()

        if self.endpoint.proxy and self.http_client is None:
            self.http_client = httpx.AsyncClient(proxy=self.endpoint.proxy)

        if self.endpoint.is_vertex or self.endpoint.endpoint_type == "anthropic_vertex":
            # Vertex needs token refresh on each access
            if self.endpoint.credentials is None:
                raise ValueError("Anthropic Vertex endpoint requires credentials")

            access_token, expires_at = get_token_with_cache(
                credentials=self.endpoint.credentials,
                proxy=self.endpoint.proxy,
                cached_token=self.endpoint.access_token,
                cached_expires_at=self.endpoint.access_token_expires_at,
            )
            self.endpoint.access_token = access_token
            self.endpoint.access_token_expires_at = expires_at

            if self.endpoint.api_base is None:
                base_url = None
            else:
                base_url = f"{self.endpoint.api_base}{self.endpoint.region}-aiplatform/v1"

            region = NOT_GIVEN if self.endpoint.region is None else self.endpoint.region
            return AsyncAnthropicVertex(
                region=region,
                base_url=base_url,
                project_id=self.endpoint.credentials.get("quota_project_id", NOT_GIVEN),
                access_token=access_token,
                http_client=self.http_client,
            )
        elif self.endpoint.is_bedrock or self.endpoint.endpoint_type == "anthropic_bedrock":
            if self.endpoint.credentials is None:
                raise ValueError("Anthropic Bedrock endpoint requires credentials")
            return AsyncAnthropicBedrock(
                aws_access_key=self.endpoint.credentials.get("access_key"),
                aws_secret_key=self.endpoint.credentials.get("secret_key"),
                aws_region=self.endpoint.region,
                http_client=self.http_client,
            )
        elif self.endpoint.endpoint_type in ("default", "anthropic"):
            if self._cached_raw_client is None:
                self._cached_raw_client = AsyncAnthropic(
                    api_key=self.endpoint.api_key,
                    base_url=self.endpoint.api_base,
                    http_client=self.http_client,
                )
            return self._cached_raw_client
        else:
            if self._cached_raw_client is None:
                self._cached_raw_client = AsyncOpenAICompatibleChatClient(
                    model=self.model,
                    stream=self.stream,
                    temperature=cast(Any, self.temperature),
                    context_length_control=self.context_length_control,
                    random_endpoint=self.random_endpoint,
                    endpoint_id=self.endpoint_id,
                    http_client=self.http_client,
                    backend_name=self.BACKEND_NAME,
                ).raw_client
            return self._cached_raw_client

    @overload
    async def create_completion(
        self,
        *,
        messages: list,
        model: str | None = None,
        stream: Literal[False] = False,
        temperature: float | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = None,
        tools: Iterable[ToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
        response_format: ResponseFormat | NotGiven = NOT_GIVEN,
        stream_options: ChatCompletionStreamOptionsParam | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        skip_cutoff: bool = False,
        audio: ChatCompletionAudioParam | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        frequency_penalty: float | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        logit_bias: dict[str, int] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        logprobs: bool | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        max_completion_tokens: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        metadata: Metadata | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        modalities: list[ChatCompletionModality] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        n: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        parallel_tool_calls: bool | OpenAINotGiven = OPENAI_NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        presence_penalty: float | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        thinking: ThinkingConfigParam | None | NotGiven = NOT_GIVEN,
        seed: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        service_tier: Literal["auto", "default"] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        stop: str | list[str] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        store: bool | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        top_logprobs: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        user: str | OpenAINotGiven = OPENAI_NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | OpenAINotGiven = OPENAI_NOT_GIVEN,
    ) -> ChatCompletionMessage:
        pass

    @overload
    async def create_completion(
        self,
        *,
        messages: list,
        model: str | None = None,
        stream: Literal[True],
        temperature: float | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = None,
        tools: Iterable[ToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
        response_format: ResponseFormat | NotGiven = NOT_GIVEN,
        stream_options: ChatCompletionStreamOptionsParam | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        skip_cutoff: bool = False,
        audio: ChatCompletionAudioParam | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        frequency_penalty: float | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        logit_bias: dict[str, int] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        logprobs: bool | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        max_completion_tokens: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        metadata: Metadata | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        modalities: list[ChatCompletionModality] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        n: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        parallel_tool_calls: bool | OpenAINotGiven = OPENAI_NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        presence_penalty: float | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        thinking: ThinkingConfigParam | None | NotGiven = NOT_GIVEN,
        seed: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        service_tier: Literal["auto", "default"] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        stop: str | list[str] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        store: bool | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        top_logprobs: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        user: str | OpenAINotGiven = OPENAI_NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | OpenAINotGiven = OPENAI_NOT_GIVEN,
    ) -> AsyncGenerator[ChatCompletionDeltaMessage, Any]:
        pass

    @overload
    async def create_completion(
        self,
        *,
        messages: list,
        model: str | None = None,
        stream: bool,
        temperature: float | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = None,
        tools: Iterable[ToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
        response_format: ResponseFormat | NotGiven = NOT_GIVEN,
        stream_options: ChatCompletionStreamOptionsParam | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        skip_cutoff: bool = False,
        audio: ChatCompletionAudioParam | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        frequency_penalty: float | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        logit_bias: dict[str, int] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        logprobs: bool | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        max_completion_tokens: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        metadata: Metadata | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        modalities: list[ChatCompletionModality] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        n: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        parallel_tool_calls: bool | OpenAINotGiven = OPENAI_NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        presence_penalty: float | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        thinking: ThinkingConfigParam | None | NotGiven = NOT_GIVEN,
        seed: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        service_tier: Literal["auto", "default"] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        stop: str | list[str] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        store: bool | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        top_logprobs: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        user: str | OpenAINotGiven = OPENAI_NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | OpenAINotGiven = OPENAI_NOT_GIVEN,
    ) -> ChatCompletionMessage | AsyncGenerator[ChatCompletionDeltaMessage, Any]:
        pass

    async def create_completion(
        self,
        *,
        messages: list,
        model: str | None = None,
        stream: Literal[False] | Literal[True] = False,
        temperature: float | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = None,
        tools: Iterable[ToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
        response_format: ResponseFormat | NotGiven = NOT_GIVEN,
        stream_options: ChatCompletionStreamOptionsParam | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        skip_cutoff: bool = False,
        audio: ChatCompletionAudioParam | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        frequency_penalty: float | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        logit_bias: dict[str, int] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        logprobs: bool | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        max_completion_tokens: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        metadata: Metadata | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        modalities: list[ChatCompletionModality] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        n: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        parallel_tool_calls: bool | OpenAINotGiven = OPENAI_NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        presence_penalty: float | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        thinking: ThinkingConfigParam | None | NotGiven = NOT_GIVEN,
        seed: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        service_tier: Literal["auto", "default"] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        stop: str | list[str] | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        store: bool | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        top_logprobs: int | OpenAINotGiven | None = OPENAI_NOT_GIVEN,
        user: str | OpenAINotGiven = OPENAI_NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | OpenAINotGiven = OPENAI_NOT_GIVEN,
    ) -> ChatCompletionMessage | AsyncGenerator[ChatCompletionDeltaMessage, Any]:
        if model is not None:
            self.model = model
        if stream is not None:
            self.stream = stream
        if temperature is not None:
            self.temperature = temperature

        self.model_setting = self.backend_settings.models[self.model]
        if self.model_id is None:
            self.model_id = self.model_setting.id

        self.endpoint, self.model_id = self._set_endpoint()

        if self.endpoint.endpoint_type and self.endpoint.endpoint_type.startswith("openai"):
            _tools = OPENAI_NOT_GIVEN if tools is NOT_GIVEN else tools
            _tool_choice = OPENAI_NOT_GIVEN if tool_choice is NOT_GIVEN else tool_choice

            formatted_messages = refactor_into_openai_messages(messages)

            if self.stream:

                async def _generator():
                    client = cast(
                        Any,
                        AsyncOpenAICompatibleChatClient(
                            model=self.model,
                            stream=True,
                            temperature=cast(Any, self.temperature),
                            context_length_control=self.context_length_control,
                            random_endpoint=self.random_endpoint,
                            endpoint_id=self.endpoint_id,
                            http_client=self.http_client,
                            backend_name=self.BACKEND_NAME,
                        ),
                    )
                    response = await client.create_completion(
                        messages=formatted_messages,
                        model=model,
                        stream=True,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=_tools,
                        tool_choice=_tool_choice,
                        response_format=response_format,
                        stream_options=stream_options,
                        top_p=top_p,
                        skip_cutoff=skip_cutoff,
                        audio=audio,
                        frequency_penalty=frequency_penalty,
                        logit_bias=logit_bias,
                        logprobs=logprobs,
                        max_completion_tokens=max_completion_tokens,
                        metadata=metadata,
                        modalities=modalities,
                        n=n,
                        parallel_tool_calls=parallel_tool_calls,
                        prediction=prediction,
                        presence_penalty=presence_penalty,
                        reasoning_effort=reasoning_effort,
                        seed=seed,
                        service_tier=service_tier,
                        stop=stop,
                        store=store,
                        top_logprobs=top_logprobs,
                        user=user,
                        extra_headers=extra_headers,
                        extra_query=extra_query,
                        extra_body=extra_body,
                        timeout=timeout,
                    )
                    async for chunk in response:
                        yield chunk

                return _generator()
            else:
                client = cast(
                    Any,
                    AsyncOpenAICompatibleChatClient(
                        model=self.model,
                        stream=False,
                        temperature=cast(Any, self.temperature),
                        context_length_control=self.context_length_control,
                        random_endpoint=self.random_endpoint,
                        endpoint_id=self.endpoint_id,
                        http_client=self.http_client,
                        backend_name=self.BACKEND_NAME,
                    ),
                )
                return await client.create_completion(
                    messages=formatted_messages,
                    model=model,
                    stream=False,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=_tools,
                    tool_choice=_tool_choice,
                    response_format=response_format,
                    top_p=top_p,
                    skip_cutoff=skip_cutoff,
                    audio=audio,
                    frequency_penalty=frequency_penalty,
                    logit_bias=logit_bias,
                    logprobs=logprobs,
                    max_completion_tokens=max_completion_tokens,
                    metadata=metadata,
                    modalities=modalities,
                    n=n,
                    parallel_tool_calls=parallel_tool_calls,
                    prediction=prediction,
                    presence_penalty=presence_penalty,
                    reasoning_effort=reasoning_effort,
                    seed=seed,
                    service_tier=service_tier,
                    stop=stop,
                    store=store,
                    top_logprobs=top_logprobs,
                    user=user,
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    extra_body=extra_body,
                    timeout=timeout,
                )

        raw_client = self.raw_client  # 调用完 self.raw_client 后，self.model_id 会被赋值
        assert isinstance(raw_client, AsyncAnthropic | AsyncAnthropicVertex | AsyncAnthropicBedrock)

        _tools = omit if isinstance(tools, NotGiven) else tools
        _tool_choice = omit if isinstance(tool_choice, NotGiven) else tool_choice
        _top_p = omit if isinstance(top_p, NotGiven) or top_p is None else top_p
        if isinstance(self.temperature, NotGiven) or self.temperature is None:
            self.temperature = omit
        _thinking = omit if isinstance(thinking, NotGiven) or thinking is None else thinking

        if messages[0].get("role") == "system":
            system_content = messages[0]["content"]
            # Preserve list format for system prompt to support cache_control
            if isinstance(system_content, list):
                system_prompt = system_content  # Keep as list to preserve cache_control
            else:
                system_prompt = system_content  # String format
            messages = messages[1:]
        else:
            system_prompt = ""

        if not skip_cutoff and self.context_length_control == ContextLengthControlType.Latest:
            messages = cutoff_messages(
                messages,
                max_count=self.model_setting.context_length,
                backend=self.BACKEND_NAME,
                model=self.model,
            )

        messages = format_messages_alternate(messages)

        tools_params: list[AnthropicToolParam] | Omit = refactor_tool_use_params(_tools) if _tools else omit
        tool_choice_param = omit
        if _tool_choice:
            tool_choice_param = refactor_tool_choice(_tool_choice)

        if not max_tokens:
            max_output_tokens = self.model_setting.max_output_tokens
            native_multimodal = self.model_setting.native_multimodal
            token_counts = get_message_token_counts(messages=messages, tools=tools, model=self.model, native_multimodal=native_multimodal)
            if max_output_tokens is not None:
                max_tokens = self.model_setting.context_length - token_counts
                max_tokens = min(max(max_tokens, 1), max_output_tokens)
            else:
                max_tokens = self.model_setting.context_length - token_counts

        await self._acquire_rate_limit(self.endpoint, self.model, messages)

        if self.stream:
            try:
                stream_response = await raw_client.messages.create(
                    model=self.model_id,
                    messages=messages,
                    system=system_prompt,
                    stream=True,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    tools=tools_params,
                    tool_choice=tool_choice_param,
                    top_p=_top_p,
                    thinking=_thinking,
                )
            except AnthropicAPIStatusError as e:
                raise APIStatusError(message=e.message, response=e.response, body=e.body) from e
            except AnthropicAPIConnectionError as e:
                raise APIConnectionError(message=e.message, request=e.request) from e

            async def generator():
                stream_state = init_anthropic_stream_state()
                async for chunk in stream_response:
                    message = adapt_anthropic_stream_event(chunk, stream_state)
                    if message is not None:
                        yield message

            return generator()
        else:
            try:
                response = await raw_client.messages.create(
                    model=self.model_id,
                    messages=messages,
                    system=system_prompt,
                    stream=False,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    tools=tools_params,
                    tool_choice=tool_choice_param,
                    top_p=_top_p,
                    thinking=_thinking,
                )
            except AnthropicAPIStatusError as e:
                raise APIStatusError(message=e.message, response=e.response, body=e.body) from e
            except AnthropicAPIConnectionError as e:
                raise APIConnectionError(message=e.message, request=e.request) from e

            return build_anthropic_completion_message(list(response.content), response.usage)
