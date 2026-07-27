from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from .types.chat_request import CapabilityPolicy, ChatRequest, ChatRequestOptions
from .types.exception import VvLlmError, classify_exception
from .types.response import CompletionResult, ResponseMetadata
from .utilities.retry_executor import RetryPolicy, execute_with_retry, execute_with_retry_async


@dataclass
class MiddlewareContext:
    provider: str | None
    model: str | None
    attempt: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)


class ChatMiddlewareV1:
    api_version = "v1"

    def on_request(self, context: MiddlewareContext, request: ChatRequest) -> ChatRequest:
        return request

    def on_response(self, context: MiddlewareContext, response: Any) -> Any:
        return response

    def on_error(self, context: MiddlewareContext, error: VvLlmError) -> None:
        return None


class MiddlewareChatClient:
    def __init__(
        self,
        client: Any,
        middleware: list[ChatMiddlewareV1] | tuple[ChatMiddlewareV1, ...] = (),
        *,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.client = client
        self.middleware = tuple(middleware)
        self.retry_policy = retry_policy or RetryPolicy(max_attempts=1)
        _validate_middleware(self.middleware)

    def create(
        self,
        request: ChatRequest,
        *,
        capability_policy: CapabilityPolicy = CapabilityPolicy.WARN,
    ) -> Any:
        response, _ = self._execute(request, capability_policy=capability_policy)
        return response

    def create_with_metadata(
        self,
        request: ChatRequest,
        *,
        capability_policy: CapabilityPolicy = CapabilityPolicy.WARN,
    ) -> CompletionResult[Any]:
        started = perf_counter()
        response, context = self._execute(request, capability_policy=capability_policy)
        return CompletionResult(
            response=response,
            metadata=ResponseMetadata(
                provider=context.provider,
                model=context.model,
                response_id=getattr(response, "id", None),
                request_id=getattr(response, "request_id", None),
                finish_reason=getattr(response, "finish_reason", None),
                attempts=max(1, context.attempt),
                latency_ms=(perf_counter() - started) * 1000,
                attributes=dict(context.attributes),
            ),
        )

    def _execute(
        self,
        request: ChatRequest,
        *,
        capability_policy: CapabilityPolicy,
    ) -> tuple[Any, MiddlewareContext]:
        context, request = self._prepare(request)

        def operation() -> Any:
            context.attempt += 1
            try:
                response = self.client.create(
                    request,
                    capability_policy=capability_policy,
                )
            except Exception as exception:
                error = classify_exception(
                    exception,
                    provider=context.provider,
                    model=context.model,
                )
                for middleware in reversed(self.middleware):
                    middleware.on_error(context, error)
                raise error from exception
            for middleware in reversed(self.middleware):
                response = middleware.on_response(context, response)
            return response

        response = execute_with_retry(
            operation,
            self.retry_policy,
            provider=context.provider,
            model=context.model,
        )
        return response, context

    def create_completion(self, **kwargs: Any) -> Any:
        return self.create(_request_from_completion_kwargs(kwargs))

    def create_stream(self, **kwargs: Any) -> Any:
        return self.create_completion(**kwargs, stream=True)

    def _prepare(self, request: ChatRequest) -> tuple[MiddlewareContext, ChatRequest]:
        provider = getattr(getattr(self.client, "backend_name", None), "value", None)
        context = MiddlewareContext(
            provider=provider,
            model=request.model or getattr(self.client, "model", None),
        )
        for middleware in self.middleware:
            request = middleware.on_request(context, request)
        context.model = request.model or context.model
        return context, request

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)


class AsyncMiddlewareChatClient:
    def __init__(
        self,
        client: Any,
        middleware: list[ChatMiddlewareV1] | tuple[ChatMiddlewareV1, ...] = (),
        *,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.client = client
        self.middleware = tuple(middleware)
        self.retry_policy = retry_policy or RetryPolicy(max_attempts=1)
        _validate_middleware(self.middleware)

    async def create(
        self,
        request: ChatRequest,
        *,
        capability_policy: CapabilityPolicy = CapabilityPolicy.WARN,
    ) -> Any:
        response, _ = await self._execute(request, capability_policy=capability_policy)
        return response

    async def create_with_metadata(
        self,
        request: ChatRequest,
        *,
        capability_policy: CapabilityPolicy = CapabilityPolicy.WARN,
    ) -> CompletionResult[Any]:
        started = perf_counter()
        response, context = await self._execute(
            request,
            capability_policy=capability_policy,
        )
        return CompletionResult(
            response=response,
            metadata=ResponseMetadata(
                provider=context.provider,
                model=context.model,
                response_id=getattr(response, "id", None),
                request_id=getattr(response, "request_id", None),
                finish_reason=getattr(response, "finish_reason", None),
                attempts=max(1, context.attempt),
                latency_ms=(perf_counter() - started) * 1000,
                attributes=dict(context.attributes),
            ),
        )

    async def _execute(
        self,
        request: ChatRequest,
        *,
        capability_policy: CapabilityPolicy,
    ) -> tuple[Any, MiddlewareContext]:
        context, request = self._prepare(request)

        async def operation() -> Any:
            context.attempt += 1
            try:
                response = await self.client.create(
                    request,
                    capability_policy=capability_policy,
                )
            except Exception as exception:
                error = classify_exception(
                    exception,
                    provider=context.provider,
                    model=context.model,
                )
                for middleware in reversed(self.middleware):
                    middleware.on_error(context, error)
                raise error from exception
            for middleware in reversed(self.middleware):
                response = middleware.on_response(context, response)
            return response

        response = await execute_with_retry_async(
            operation,
            self.retry_policy,
            provider=context.provider,
            model=context.model,
        )
        return response, context

    async def create_completion(self, **kwargs: Any) -> Any:
        return await self.create(_request_from_completion_kwargs(kwargs))

    async def create_stream(self, **kwargs: Any) -> Any:
        return await self.create_completion(**kwargs, stream=True)

    def _prepare(self, request: ChatRequest) -> tuple[MiddlewareContext, ChatRequest]:
        provider = getattr(getattr(self.client, "backend_name", None), "value", None)
        context = MiddlewareContext(
            provider=provider,
            model=request.model or getattr(self.client, "model", None),
        )
        for middleware in self.middleware:
            request = middleware.on_request(context, request)
        context.model = request.model or context.model
        return context, request

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)


def _validate_middleware(middleware: tuple[ChatMiddlewareV1, ...]) -> None:
    for item in middleware:
        if item.api_version != ChatMiddlewareV1.api_version:
            raise ValueError(f"unsupported middleware API version: {item.api_version}")


def _request_from_completion_kwargs(values: dict[str, Any]) -> ChatRequest:
    kwargs = dict(values)
    messages = kwargs.pop("messages")
    model = kwargs.pop("model", None)
    stream = bool(kwargs.pop("stream", False))
    tools = kwargs.pop("tools", None)
    tool_choice = kwargs.pop("tool_choice", None)
    skip_cutoff = bool(kwargs.pop("skip_cutoff", False))
    extra_headers = kwargs.pop("extra_headers", None)
    header_context = kwargs.pop("header_context", None)
    extra_query = kwargs.pop("extra_query", None)
    extra_body = kwargs.pop("extra_body", None)
    option_fields = ChatRequestOptions.model_fields
    options = {key: kwargs.pop(key) for key in tuple(kwargs) if key in option_fields}
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unsupported completion arguments: {unknown}")
    return ChatRequest(
        messages=messages,
        model=model,
        stream=stream,
        options=ChatRequestOptions(**options),
        tools=tools,
        tool_choice=tool_choice,
        skip_cutoff=skip_cutoff,
        extra_headers=extra_headers,
        header_context=header_context,
        extra_query=extra_query,
        extra_body=extra_body,
    )


__all__ = [
    "AsyncMiddlewareChatClient",
    "ChatMiddlewareV1",
    "MiddlewareChatClient",
    "MiddlewareContext",
]
