import time
import random
import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Generator, AsyncGenerator
from functools import cached_property
from typing import Any, overload, Literal

import httpx
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
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
    AsyncAnthropic,
    AnthropicVertex,
    AsyncAnthropicVertex,
    AnthropicBedrock,
    AsyncAnthropicBedrock,
)
from anthropic.types.thinking_config_param import ThinkingConfigParam

from ..settings import Settings, normalize_settings
from ..types import defaults as defs
from ..types.settings import SettingsDict
from ..types.enums import ContextLengthControlType, BackendType
from ..types.llm_parameters import (
    NotGiven,
    NOT_GIVEN,
    ToolParam,
    ToolChoice,
    OpenAINotGiven,
    EndpointSetting,
    ChatCompletionMessage,
    ChatCompletionDeltaMessage,
)
from ..utilities.rate_limiter import SyncMemoryRateLimiter, SyncRedisRateLimiter, SyncDiskCacheRateLimiter
from ..utilities.rate_limiter import AsyncMemoryRateLimiter, AsyncRedisRateLimiter, AsyncDiskCacheRateLimiter


class BaseChatClient(ABC):
    DEFAULT_MODEL: str
    BACKEND_NAME: BackendType

    def __init__(
        self,
        model: str = "",
        stream: bool = False,
        temperature: float | None | NotGiven = NOT_GIVEN,
        context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
        random_endpoint: bool = True,
        endpoint_id: str = "",
        http_client: httpx.Client | None = None,
        backend_name: str | None = None,
        settings: Settings | SettingsDict | None = None,  # Use default settings if not provided
    ):
        self.model = model or self.DEFAULT_MODEL
        self.stream = stream
        self.temperature = temperature
        self.context_length_control = context_length_control
        self.random_endpoint = random_endpoint
        self.endpoint_id = endpoint_id
        self.http_client = http_client

        if backend_name is not None:
            self.backend_name = BackendType(backend_name)
        else:
            self.backend_name = self.BACKEND_NAME

        self.settings: Settings = normalize_settings(settings)

        self.backend_settings = self.settings.get_backend(self.backend_name)

        self.rate_limiter = self._init_rate_limiter()
        self.active_requests = defaultdict(int)
        self.rpm = None
        self.tpm = None
        self.concurrent_requests = None

        if endpoint_id:
            self.endpoint_id = endpoint_id
            self.random_endpoint = False
            self.endpoint = self.settings.get_endpoint(self.endpoint_id)

    def _init_rate_limiter(self):
        if not self.settings.rate_limit:
            return None
        if not self.settings.rate_limit.enabled:
            return None

        if self.settings.rate_limit.backend == "memory":
            return SyncMemoryRateLimiter()
        elif self.settings.rate_limit.backend == "redis":
            if not self.settings.rate_limit.redis:
                raise ValueError("Redis settings must be provided if Redis backend is selected.")
            return SyncRedisRateLimiter(
                host=self.settings.rate_limit.redis.host,
                port=self.settings.rate_limit.redis.port,
                db=self.settings.rate_limit.redis.db,
            )
        elif self.settings.rate_limit.backend == "diskcache":
            if not self.settings.rate_limit.diskcache:
                raise ValueError("Diskcache settings must be provided if Diskcache backend is selected.")
            return SyncDiskCacheRateLimiter(
                cache_dir=self.settings.rate_limit.diskcache.cache_dir,
            )
        return None

    def _acquire_rate_limit(self, endpoint: EndpointSetting | None, model: str, messages: list):
        if endpoint is None:
            return

        key = f"{endpoint.id}:{model}"

        # Get rate limit parameters
        # Priority: parameters in model.endpoints > parameters in endpoint > default parameters
        rpm = self.rpm or endpoint.rpm or (self.settings.rate_limit.default_rpm if self.settings.rate_limit else 60)
        tpm = self.tpm or endpoint.tpm or (self.settings.rate_limit.default_tpm if self.settings.rate_limit else 1000000)

        while self.rate_limiter:
            allowed, wait_time = self.rate_limiter.check_limit(key, rpm, tpm, self._estimate_request_tokens(messages))
            if allowed:
                break
            time.sleep(wait_time)

    def _estimate_request_tokens(self, messages: list) -> int:
        """Roughly estimate the number of tokens in the request.
        Uses higher ratio for CJK characters which typically consume more tokens.
        """
        tokens = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                cjk_chars = sum(1 for c in content if '\u4e00' <= c <= '\u9fff' or '\u3000' <= c <= '\u303f' or '\uff00' <= c <= '\uffef')
                ascii_chars = len(content) - cjk_chars
                tokens += int(cjk_chars * 1.5 + ascii_chars * 0.25)
        return max(tokens, 1)

    def _get_available_endpoints(self, model_endpoints: list) -> list:
        """Get list of available (enabled) endpoints for the model"""
        available_endpoints = []
        for endpoint_option in model_endpoints:
            if isinstance(endpoint_option, dict):
                endpoint_id = endpoint_option["endpoint_id"]
                try:
                    endpoint = self.settings.get_endpoint(endpoint_id)
                    if endpoint.enabled:
                        available_endpoints.append(endpoint_option)
                except ValueError:
                    continue
            else:
                try:
                    endpoint = self.settings.get_endpoint(endpoint_option)
                    if endpoint.enabled:
                        available_endpoints.append(endpoint_option)
                except ValueError:
                    continue
        return available_endpoints

    def set_model_id_by_endpoint_id(self, endpoint_id: str):
        for endpoint_option in self.backend_settings.models[self.model].endpoints:
            if isinstance(endpoint_option, dict) and endpoint_id == endpoint_option["endpoint_id"]:
                self.model_id = endpoint_option["model_id"]
                break
        return self.model_id

    def _set_endpoint(self):
        if self.endpoint is None:
            if self.random_endpoint:
                available_endpoints = self._get_available_endpoints(self.backend_settings.models[self.model].endpoints)
                if not available_endpoints:
                    raise ValueError(f"No enabled endpoints available for model {self.model}")

                endpoint = random.choice(available_endpoints)
                if isinstance(endpoint, dict):
                    self.endpoint_id = endpoint["endpoint_id"]
                    self.model_id = endpoint["model_id"]
                    self.rpm = endpoint.get("rpm", None)
                    self.tpm = endpoint.get("tpm", None)
                    self.concurrent_requests = endpoint.get("concurrent_requests", None)
                else:
                    self.endpoint_id = endpoint
                self.endpoint = self.settings.get_endpoint(self.endpoint_id)
            else:
                self.endpoint = self.settings.get_endpoint(self.endpoint_id)
                if not self.endpoint.enabled:
                    raise ValueError(f"Endpoint {self.endpoint_id} is disabled")
                self.set_model_id_by_endpoint_id(self.endpoint_id)
        elif isinstance(self.endpoint, EndpointSetting):
            if not self.endpoint.enabled:
                raise ValueError(f"Endpoint {self.endpoint.id} is disabled")
            self.endpoint_id = self.endpoint.id
            self.set_model_id_by_endpoint_id(self.endpoint_id)
        else:
            raise ValueError("Invalid endpoint")

        return self.endpoint, self.model_id

    @cached_property
    @abstractmethod
    def raw_client(
        self,
    ) -> OpenAI | AzureOpenAI | Anthropic | AnthropicVertex | AnthropicBedrock | httpx.Client | None:
        pass

    @overload
    @abstractmethod
    def create_completion(
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
        audio: ChatCompletionAudioParam | OpenAINotGiven | None = NOT_GIVEN,
        frequency_penalty: float | OpenAINotGiven | None = NOT_GIVEN,
        logit_bias: dict[str, int] | OpenAINotGiven | None = NOT_GIVEN,
        logprobs: bool | OpenAINotGiven | None = NOT_GIVEN,
        max_completion_tokens: int | OpenAINotGiven | None = NOT_GIVEN,
        metadata: Metadata | OpenAINotGiven | None = NOT_GIVEN,
        modalities: list[ChatCompletionModality] | OpenAINotGiven | None = NOT_GIVEN,
        n: int | OpenAINotGiven | None = NOT_GIVEN,
        parallel_tool_calls: bool | OpenAINotGiven = NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam | OpenAINotGiven | None = NOT_GIVEN,
        presence_penalty: float | OpenAINotGiven | None = NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | OpenAINotGiven | None = NOT_GIVEN,
        thinking: ThinkingConfigParam | None | NotGiven = NOT_GIVEN,
        seed: int | OpenAINotGiven | None = NOT_GIVEN,
        service_tier: Literal["auto", "default"] | OpenAINotGiven | None = NOT_GIVEN,
        stop: str | list[str] | OpenAINotGiven | None = NOT_GIVEN,
        store: bool | OpenAINotGiven | None = NOT_GIVEN,
        top_logprobs: int | OpenAINotGiven | None = NOT_GIVEN,
        user: str | OpenAINotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | OpenAINotGiven = NOT_GIVEN,
    ) -> ChatCompletionMessage:
        pass

    @overload
    @abstractmethod
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
        audio: ChatCompletionAudioParam | OpenAINotGiven | None = NOT_GIVEN,
        frequency_penalty: float | OpenAINotGiven | None = NOT_GIVEN,
        logit_bias: dict[str, int] | OpenAINotGiven | None = NOT_GIVEN,
        logprobs: bool | OpenAINotGiven | None = NOT_GIVEN,
        max_completion_tokens: int | OpenAINotGiven | None = NOT_GIVEN,
        metadata: Metadata | OpenAINotGiven | None = NOT_GIVEN,
        modalities: list[ChatCompletionModality] | OpenAINotGiven | None = NOT_GIVEN,
        n: int | OpenAINotGiven | None = NOT_GIVEN,
        parallel_tool_calls: bool | OpenAINotGiven = NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam | OpenAINotGiven | None = NOT_GIVEN,
        presence_penalty: float | OpenAINotGiven | None = NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | OpenAINotGiven | None = NOT_GIVEN,
        thinking: ThinkingConfigParam | None | NotGiven = NOT_GIVEN,
        seed: int | OpenAINotGiven | None = NOT_GIVEN,
        service_tier: Literal["auto", "default"] | OpenAINotGiven | None = NOT_GIVEN,
        stop: str | list[str] | OpenAINotGiven | None = NOT_GIVEN,
        store: bool | OpenAINotGiven | None = NOT_GIVEN,
        top_logprobs: int | OpenAINotGiven | None = NOT_GIVEN,
        user: str | OpenAINotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | OpenAINotGiven = NOT_GIVEN,
    ) -> Generator[ChatCompletionDeltaMessage, Any, None]:
        pass

    @overload
    @abstractmethod
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
        audio: ChatCompletionAudioParam | OpenAINotGiven | None = NOT_GIVEN,
        frequency_penalty: float | OpenAINotGiven | None = NOT_GIVEN,
        logit_bias: dict[str, int] | OpenAINotGiven | None = NOT_GIVEN,
        logprobs: bool | OpenAINotGiven | None = NOT_GIVEN,
        max_completion_tokens: int | OpenAINotGiven | None = NOT_GIVEN,
        metadata: Metadata | OpenAINotGiven | None = NOT_GIVEN,
        modalities: list[ChatCompletionModality] | OpenAINotGiven | None = NOT_GIVEN,
        n: int | OpenAINotGiven | None = NOT_GIVEN,
        parallel_tool_calls: bool | OpenAINotGiven = NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam | OpenAINotGiven | None = NOT_GIVEN,
        presence_penalty: float | OpenAINotGiven | None = NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | OpenAINotGiven | None = NOT_GIVEN,
        thinking: ThinkingConfigParam | None | NotGiven = NOT_GIVEN,
        seed: int | OpenAINotGiven | None = NOT_GIVEN,
        service_tier: Literal["auto", "default"] | OpenAINotGiven | None = NOT_GIVEN,
        stop: str | list[str] | OpenAINotGiven | None = NOT_GIVEN,
        store: bool | OpenAINotGiven | None = NOT_GIVEN,
        top_logprobs: int | OpenAINotGiven | None = NOT_GIVEN,
        user: str | OpenAINotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | OpenAINotGiven = NOT_GIVEN,
    ) -> ChatCompletionMessage | Generator[ChatCompletionDeltaMessage, Any, None]:
        pass

    @abstractmethod
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
        audio: ChatCompletionAudioParam | OpenAINotGiven | None = NOT_GIVEN,
        frequency_penalty: float | OpenAINotGiven | None = NOT_GIVEN,
        logit_bias: dict[str, int] | OpenAINotGiven | None = NOT_GIVEN,
        logprobs: bool | OpenAINotGiven | None = NOT_GIVEN,
        max_completion_tokens: int | OpenAINotGiven | None = NOT_GIVEN,
        metadata: Metadata | OpenAINotGiven | None = NOT_GIVEN,
        modalities: list[ChatCompletionModality] | OpenAINotGiven | None = NOT_GIVEN,
        n: int | OpenAINotGiven | None = NOT_GIVEN,
        parallel_tool_calls: bool | OpenAINotGiven = NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam | OpenAINotGiven | None = NOT_GIVEN,
        presence_penalty: float | OpenAINotGiven | None = NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | OpenAINotGiven | None = NOT_GIVEN,
        thinking: ThinkingConfigParam | None | NotGiven = NOT_GIVEN,
        seed: int | OpenAINotGiven | None = NOT_GIVEN,
        service_tier: Literal["auto", "default"] | OpenAINotGiven | None = NOT_GIVEN,
        stop: str | list[str] | OpenAINotGiven | None = NOT_GIVEN,
        store: bool | OpenAINotGiven | None = NOT_GIVEN,
        top_logprobs: int | OpenAINotGiven | None = NOT_GIVEN,
        user: str | OpenAINotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | OpenAINotGiven = NOT_GIVEN,
    ) -> ChatCompletionMessage | Generator[ChatCompletionDeltaMessage, Any, None]:
        pass

    def create_stream(
        self,
        *,
        messages: list,
        model: str | None = None,
        temperature: float | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = None,
        tools: Iterable[ToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
        response_format: ResponseFormat | NotGiven = NOT_GIVEN,
        stream_options: ChatCompletionStreamOptionsParam | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        skip_cutoff: bool = False,
        audio: ChatCompletionAudioParam | OpenAINotGiven | None = NOT_GIVEN,
        frequency_penalty: float | OpenAINotGiven | None = NOT_GIVEN,
        logit_bias: dict[str, int] | OpenAINotGiven | None = NOT_GIVEN,
        logprobs: bool | OpenAINotGiven | None = NOT_GIVEN,
        max_completion_tokens: int | OpenAINotGiven | None = NOT_GIVEN,
        metadata: Metadata | OpenAINotGiven | None = NOT_GIVEN,
        modalities: list[ChatCompletionModality] | OpenAINotGiven | None = NOT_GIVEN,
        n: int | OpenAINotGiven | None = NOT_GIVEN,
        parallel_tool_calls: bool | OpenAINotGiven = NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam | OpenAINotGiven | None = NOT_GIVEN,
        presence_penalty: float | OpenAINotGiven | None = NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | OpenAINotGiven | None = NOT_GIVEN,
        thinking: ThinkingConfigParam | None | NotGiven = NOT_GIVEN,
        seed: int | OpenAINotGiven | None = NOT_GIVEN,
        service_tier: Literal["auto", "default"] | OpenAINotGiven | None = NOT_GIVEN,
        stop: str | list[str] | OpenAINotGiven | None = NOT_GIVEN,
        store: bool | OpenAINotGiven | None = NOT_GIVEN,
        top_logprobs: int | OpenAINotGiven | None = NOT_GIVEN,
        user: str | OpenAINotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | OpenAINotGiven = NOT_GIVEN,
    ) -> Generator[ChatCompletionDeltaMessage, Any, None]:
        return self.create_completion(
            messages=messages,
            model=model,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
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
            thinking=thinking,
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

    def model_list(self):
        _raw_client = self.raw_client
        if isinstance(_raw_client, OpenAI | AzureOpenAI):
            return _raw_client.models.list().model_dump()
        elif isinstance(_raw_client, Anthropic):
            return _raw_client.models.list(limit=1000).model_dump()
        else:
            raise ValueError(f"Unsupported client type: {type(_raw_client)}")


class BaseAsyncChatClient(ABC):
    DEFAULT_MODEL: str
    BACKEND_NAME: BackendType

    def __init__(
        self,
        model: str = "",
        stream: bool = False,
        temperature: float | None | NotGiven = NOT_GIVEN,
        context_length_control: ContextLengthControlType = defs.CONTEXT_LENGTH_CONTROL,
        random_endpoint: bool = True,
        endpoint_id: str = "",
        http_client: httpx.AsyncClient | None = None,
        backend_name: str | None = None,
        settings: Settings | SettingsDict | None = None,  # Use default settings if not provided
    ):
        self.model = model or self.DEFAULT_MODEL
        self.stream = stream
        self.temperature = temperature
        self.context_length_control = context_length_control
        self.random_endpoint = random_endpoint
        self.endpoint_id = endpoint_id
        self.http_client = http_client

        if backend_name is not None:
            self.backend_name = BackendType(backend_name)
        else:
            self.backend_name = self.BACKEND_NAME

        self.settings: Settings = normalize_settings(settings)

        self.backend_settings = self.settings.get_backend(self.backend_name)

        self.rate_limiter = self._init_rate_limiter()
        self.active_requests = defaultdict(int)
        self.rpm = None
        self.tpm = None
        self.concurrent_requests = None

        if endpoint_id:
            self.endpoint_id = endpoint_id
            self.random_endpoint = False
            self.endpoint = self.settings.get_endpoint(self.endpoint_id)

    def _init_rate_limiter(self):
        if not self.settings.rate_limit:
            return None
        if not self.settings.rate_limit.enabled:
            return None

        if self.settings.rate_limit.backend == "memory":
            return AsyncMemoryRateLimiter()
        elif self.settings.rate_limit.backend == "redis":
            if not self.settings.rate_limit.redis:
                raise ValueError("Redis settings must be provided if Redis backend is selected.")
            return AsyncRedisRateLimiter(
                host=self.settings.rate_limit.redis.host,
                port=self.settings.rate_limit.redis.port,
                db=self.settings.rate_limit.redis.db,
            )
        elif self.settings.rate_limit.backend == "diskcache":
            if not self.settings.rate_limit.diskcache:
                raise ValueError("Diskcache settings must be provided if Diskcache backend is selected.")
            return AsyncDiskCacheRateLimiter(
                cache_dir=self.settings.rate_limit.diskcache.cache_dir,
            )
        return None

    async def _acquire_rate_limit(self, endpoint: EndpointSetting | None, model: str, messages: list):
        if endpoint is None:
            return

        key = f"{endpoint.id}:{model}"

        # Get rate limit parameters
        # Priority: parameters in model.endpoints > parameters in endpoint > default parameters
        rpm = self.rpm or endpoint.rpm or (self.settings.rate_limit.default_rpm if self.settings.rate_limit else 60)
        tpm = self.tpm or endpoint.tpm or (self.settings.rate_limit.default_tpm if self.settings.rate_limit else 1000000)

        while self.rate_limiter:
            allowed, wait_time = await self.rate_limiter.check_limit(key, rpm, tpm, self._estimate_request_tokens(messages))
            if allowed:
                break
            await asyncio.sleep(wait_time)

    def _estimate_request_tokens(self, messages: list) -> int:
        """Roughly estimate the number of tokens in the request.
        Uses higher ratio for CJK characters which typically consume more tokens.
        """
        tokens = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                cjk_chars = sum(1 for c in content if '\u4e00' <= c <= '\u9fff' or '\u3000' <= c <= '\u303f' or '\uff00' <= c <= '\uffef')
                ascii_chars = len(content) - cjk_chars
                tokens += int(cjk_chars * 1.5 + ascii_chars * 0.25)
        return max(tokens, 1)

    def _get_available_endpoints(self, model_endpoints: list) -> list:
        """Get list of available (enabled) endpoints for the model"""
        available_endpoints = []
        for endpoint_option in model_endpoints:
            if isinstance(endpoint_option, dict):
                # For endpoint with specific config, check if the endpoint is enabled
                endpoint_id = endpoint_option["endpoint_id"]
                try:
                    endpoint = self.settings.get_endpoint(endpoint_id)
                    if endpoint.enabled:
                        available_endpoints.append(endpoint_option)
                except ValueError:
                    # Endpoint not found, skip it
                    continue
            else:
                # For simple endpoint ID string, check if the endpoint is enabled
                try:
                    endpoint = self.settings.get_endpoint(endpoint_option)
                    if endpoint.enabled:
                        available_endpoints.append(endpoint_option)
                except ValueError:
                    # Endpoint not found, skip it
                    continue
        return available_endpoints

    def set_model_id_by_endpoint_id(self, endpoint_id: str):
        for endpoint_option in self.backend_settings.models[self.model].endpoints:
            if isinstance(endpoint_option, dict) and endpoint_id == endpoint_option["endpoint_id"]:
                self.model_id = endpoint_option["model_id"]
                break
        return self.model_id

    def _set_endpoint(self):
        if self.endpoint is None:
            if self.random_endpoint:
                # Get available (enabled) endpoints
                available_endpoints = self._get_available_endpoints(self.backend_settings.models[self.model].endpoints)
                if not available_endpoints:
                    raise ValueError(f"No enabled endpoints available for model {self.model}")

                endpoint = random.choice(available_endpoints)
                if isinstance(endpoint, dict):
                    self.endpoint_id = endpoint["endpoint_id"]
                    self.model_id = endpoint["model_id"]
                    self.rpm = endpoint.get("rpm", None)
                    self.tpm = endpoint.get("tpm", None)
                    self.concurrent_requests = endpoint.get("concurrent_requests", None)
                else:
                    self.endpoint_id = endpoint
                self.endpoint = self.settings.get_endpoint(self.endpoint_id)
            else:
                self.endpoint = self.settings.get_endpoint(self.endpoint_id)
                # Check if the specified endpoint is enabled
                if not self.endpoint.enabled:
                    raise ValueError(f"Endpoint {self.endpoint_id} is disabled")
                self.set_model_id_by_endpoint_id(self.endpoint_id)
        elif isinstance(self.endpoint, EndpointSetting):
            # Check if the endpoint is enabled
            if not self.endpoint.enabled:
                raise ValueError(f"Endpoint {self.endpoint.id} is disabled")
            self.endpoint_id = self.endpoint.id
            self.set_model_id_by_endpoint_id(self.endpoint_id)
        else:
            raise ValueError("Invalid endpoint")

        return self.endpoint, self.model_id

    @cached_property
    @abstractmethod
    def raw_client(
        self,
    ) -> AsyncOpenAI | AsyncAzureOpenAI | AsyncAnthropic | AsyncAnthropicVertex | AsyncAnthropicBedrock | httpx.AsyncClient | None:
        pass

    @overload
    @abstractmethod
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
        audio: ChatCompletionAudioParam | OpenAINotGiven | None = NOT_GIVEN,
        frequency_penalty: float | OpenAINotGiven | None = NOT_GIVEN,
        logit_bias: dict[str, int] | OpenAINotGiven | None = NOT_GIVEN,
        logprobs: bool | OpenAINotGiven | None = NOT_GIVEN,
        max_completion_tokens: int | OpenAINotGiven | None = NOT_GIVEN,
        metadata: Metadata | OpenAINotGiven | None = NOT_GIVEN,
        modalities: list[ChatCompletionModality] | OpenAINotGiven | None = NOT_GIVEN,
        n: int | OpenAINotGiven | None = NOT_GIVEN,
        parallel_tool_calls: bool | OpenAINotGiven = NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam | OpenAINotGiven | None = NOT_GIVEN,
        presence_penalty: float | OpenAINotGiven | None = NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | OpenAINotGiven | None = NOT_GIVEN,
        thinking: ThinkingConfigParam | None | NotGiven = NOT_GIVEN,
        seed: int | OpenAINotGiven | None = NOT_GIVEN,
        service_tier: Literal["auto", "default"] | OpenAINotGiven | None = NOT_GIVEN,
        stop: str | list[str] | OpenAINotGiven | None = NOT_GIVEN,
        store: bool | OpenAINotGiven | None = NOT_GIVEN,
        top_logprobs: int | OpenAINotGiven | None = NOT_GIVEN,
        user: str | OpenAINotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | OpenAINotGiven = NOT_GIVEN,
    ) -> ChatCompletionMessage:
        pass

    @overload
    @abstractmethod
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
        audio: ChatCompletionAudioParam | OpenAINotGiven | None = NOT_GIVEN,
        frequency_penalty: float | OpenAINotGiven | None = NOT_GIVEN,
        logit_bias: dict[str, int] | OpenAINotGiven | None = NOT_GIVEN,
        logprobs: bool | OpenAINotGiven | None = NOT_GIVEN,
        max_completion_tokens: int | OpenAINotGiven | None = NOT_GIVEN,
        metadata: Metadata | OpenAINotGiven | None = NOT_GIVEN,
        modalities: list[ChatCompletionModality] | OpenAINotGiven | None = NOT_GIVEN,
        n: int | OpenAINotGiven | None = NOT_GIVEN,
        parallel_tool_calls: bool | OpenAINotGiven = NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam | OpenAINotGiven | None = NOT_GIVEN,
        presence_penalty: float | OpenAINotGiven | None = NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | OpenAINotGiven | None = NOT_GIVEN,
        thinking: ThinkingConfigParam | None | NotGiven = NOT_GIVEN,
        seed: int | OpenAINotGiven | None = NOT_GIVEN,
        service_tier: Literal["auto", "default"] | OpenAINotGiven | None = NOT_GIVEN,
        stop: str | list[str] | OpenAINotGiven | None = NOT_GIVEN,
        store: bool | OpenAINotGiven | None = NOT_GIVEN,
        top_logprobs: int | OpenAINotGiven | None = NOT_GIVEN,
        user: str | OpenAINotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | OpenAINotGiven = NOT_GIVEN,
    ) -> AsyncGenerator[ChatCompletionDeltaMessage, Any]:
        pass

    @overload
    @abstractmethod
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
        audio: ChatCompletionAudioParam | OpenAINotGiven | None = NOT_GIVEN,
        frequency_penalty: float | OpenAINotGiven | None = NOT_GIVEN,
        logit_bias: dict[str, int] | OpenAINotGiven | None = NOT_GIVEN,
        logprobs: bool | OpenAINotGiven | None = NOT_GIVEN,
        max_completion_tokens: int | OpenAINotGiven | None = NOT_GIVEN,
        metadata: Metadata | OpenAINotGiven | None = NOT_GIVEN,
        modalities: list[ChatCompletionModality] | OpenAINotGiven | None = NOT_GIVEN,
        n: int | OpenAINotGiven | None = NOT_GIVEN,
        parallel_tool_calls: bool | OpenAINotGiven = NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam | OpenAINotGiven | None = NOT_GIVEN,
        presence_penalty: float | OpenAINotGiven | None = NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | OpenAINotGiven | None = NOT_GIVEN,
        thinking: ThinkingConfigParam | None | NotGiven = NOT_GIVEN,
        seed: int | OpenAINotGiven | None = NOT_GIVEN,
        service_tier: Literal["auto", "default"] | OpenAINotGiven | None = NOT_GIVEN,
        stop: str | list[str] | OpenAINotGiven | None = NOT_GIVEN,
        store: bool | OpenAINotGiven | None = NOT_GIVEN,
        top_logprobs: int | OpenAINotGiven | None = NOT_GIVEN,
        user: str | OpenAINotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | OpenAINotGiven = NOT_GIVEN,
    ) -> ChatCompletionMessage | AsyncGenerator[ChatCompletionDeltaMessage, Any]:
        pass

    @abstractmethod
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
        audio: ChatCompletionAudioParam | OpenAINotGiven | None = NOT_GIVEN,
        frequency_penalty: float | OpenAINotGiven | None = NOT_GIVEN,
        logit_bias: dict[str, int] | OpenAINotGiven | None = NOT_GIVEN,
        logprobs: bool | OpenAINotGiven | None = NOT_GIVEN,
        max_completion_tokens: int | OpenAINotGiven | None = NOT_GIVEN,
        metadata: Metadata | OpenAINotGiven | None = NOT_GIVEN,
        modalities: list[ChatCompletionModality] | OpenAINotGiven | None = NOT_GIVEN,
        n: int | OpenAINotGiven | None = NOT_GIVEN,
        parallel_tool_calls: bool | OpenAINotGiven = NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam | OpenAINotGiven | None = NOT_GIVEN,
        presence_penalty: float | OpenAINotGiven | None = NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | OpenAINotGiven | None = NOT_GIVEN,
        thinking: ThinkingConfigParam | None | NotGiven = NOT_GIVEN,
        seed: int | OpenAINotGiven | None = NOT_GIVEN,
        service_tier: Literal["auto", "default"] | OpenAINotGiven | None = NOT_GIVEN,
        stop: str | list[str] | OpenAINotGiven | None = NOT_GIVEN,
        store: bool | OpenAINotGiven | None = NOT_GIVEN,
        top_logprobs: int | OpenAINotGiven | None = NOT_GIVEN,
        user: str | OpenAINotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | OpenAINotGiven = NOT_GIVEN,
    ) -> ChatCompletionMessage | AsyncGenerator[ChatCompletionDeltaMessage, Any]:
        pass

    async def create_stream(
        self,
        *,
        messages: list,
        model: str | None = None,
        temperature: float | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = None,
        tools: Iterable[ToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
        response_format: ResponseFormat | NotGiven = NOT_GIVEN,
        stream_options: ChatCompletionStreamOptionsParam | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        skip_cutoff: bool = False,
        audio: ChatCompletionAudioParam | OpenAINotGiven | None = NOT_GIVEN,
        frequency_penalty: float | OpenAINotGiven | None = NOT_GIVEN,
        logit_bias: dict[str, int] | OpenAINotGiven | None = NOT_GIVEN,
        logprobs: bool | OpenAINotGiven | None = NOT_GIVEN,
        max_completion_tokens: int | OpenAINotGiven | None = NOT_GIVEN,
        metadata: Metadata | OpenAINotGiven | None = NOT_GIVEN,
        modalities: list[ChatCompletionModality] | OpenAINotGiven | None = NOT_GIVEN,
        n: int | OpenAINotGiven | None = NOT_GIVEN,
        parallel_tool_calls: bool | OpenAINotGiven = NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam | OpenAINotGiven | None = NOT_GIVEN,
        presence_penalty: float | OpenAINotGiven | None = NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | OpenAINotGiven | None = NOT_GIVEN,
        thinking: ThinkingConfigParam | None | NotGiven = NOT_GIVEN,
        seed: int | OpenAINotGiven | None = NOT_GIVEN,
        service_tier: Literal["auto", "default"] | OpenAINotGiven | None = NOT_GIVEN,
        stop: str | list[str] | OpenAINotGiven | None = NOT_GIVEN,
        store: bool | OpenAINotGiven | None = NOT_GIVEN,
        top_logprobs: int | OpenAINotGiven | None = NOT_GIVEN,
        user: str | OpenAINotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | OpenAINotGiven = NOT_GIVEN,
    ) -> AsyncGenerator[ChatCompletionDeltaMessage, Any]:
        return await self.create_completion(
            messages=messages,
            model=model,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
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
            thinking=thinking,
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

    async def model_list(self):
        _raw_client = self.raw_client
        if isinstance(_raw_client, AsyncOpenAI | AsyncAzureOpenAI):
            return (await _raw_client.models.list()).model_dump()
        elif isinstance(_raw_client, AsyncAnthropic):
            return (await _raw_client.models.list(limit=1000)).model_dump()
        else:
            raise ValueError(f"Unsupported client type: {type(_raw_client)}")
