from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Generator, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .types.chat_request import CapabilityPolicy, ChatRequest, ModelCapabilities
from .types.exception import ErrorKind, VvLlmError, classify_exception
from .types.response import CompletionResult, ResponseMetadata


@dataclass(frozen=True)
class ProviderRegistration:
    name: str
    factory: Callable[[], Any]
    capabilities: ModelCapabilities


@dataclass(frozen=True)
class FallbackRoute:
    provider: str
    model: str


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, ProviderRegistration] = {}

    def register(
        self,
        name: str,
        factory: Callable[[], Any],
        *,
        capabilities: ModelCapabilities,
        replace: bool = False,
    ) -> None:
        if name in self._providers and not replace:
            raise ValueError(f"provider is already registered: {name}")
        self._providers[name] = ProviderRegistration(name, factory, capabilities)

    def get(self, name: str) -> ProviderRegistration:
        try:
            return self._providers[name]
        except KeyError as exception:
            raise VvLlmError(
                f"provider is not registered: {name}",
                kind=ErrorKind.CONFIGURATION,
                provider=name,
                source=exception,
            ) from exception

    def create(self, name: str) -> Any:
        return self.get(name).factory()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._providers)


class FallbackChatClient:
    def __init__(
        self,
        registry: ProviderRegistry,
        routes: list[FallbackRoute] | tuple[FallbackRoute, ...],
        *,
        fallback_on: frozenset[ErrorKind] | None = None,
    ) -> None:
        if not routes:
            raise ValueError("fallback routes cannot be empty")
        self.registry = registry
        self.routes = tuple(routes)
        self.fallback_on = fallback_on or _default_fallback_errors()

    def create(self, request: ChatRequest) -> Any:
        if request.stream:
            stream, _ = self._prepare_stream(request)
            return stream
        response, _, _ = self._complete(request)
        return response

    def create_with_metadata(self, request: ChatRequest) -> CompletionResult[Any]:
        started = perf_counter()
        if request.stream:
            response, index = self._prepare_stream(request)
            route = self.routes[index]
        else:
            response, index, route = self._complete(request)
        return CompletionResult(
            response=response,
            metadata=_metadata(response, route, index, started),
        )

    def _complete(self, request: ChatRequest) -> tuple[Any, int, FallbackRoute]:
        last_error: VvLlmError | None = None
        for index, route in enumerate(self.routes):
            registration = self.registry.get(route.provider)
            routed = request.model_copy(update={"model": route.model})
            capability_error = _capability_error(registration, routed)
            if capability_error is not None:
                last_error = capability_error
                continue
            try:
                return registration.factory().create(routed), index, route
            except Exception as exception:
                error = classify_exception(
                    exception,
                    provider=route.provider,
                    model=route.model,
                )
                if error.kind not in self.fallback_on or index == len(self.routes) - 1:
                    raise error from exception
                last_error = error
        raise last_error or VvLlmError(
            "no fallback route was eligible",
            kind=ErrorKind.CONFIGURATION,
        )

    def _prepare_stream(self, request: ChatRequest) -> tuple[Generator[Any, None, None], int]:
        last_error: VvLlmError | None = None
        for index, route in enumerate(self.routes):
            registration = self.registry.get(route.provider)
            routed = request.model_copy(update={"model": route.model, "stream": True})
            capability_error = _capability_error(registration, routed)
            if capability_error is not None:
                last_error = capability_error
                continue
            try:
                iterator = iter(registration.factory().create(routed))
                prelude: list[Any] = []
                while True:
                    try:
                        chunk = next(iterator)
                    except StopIteration:
                        # A route may legitimately finish with only usage,
                        # done, or empty deltas.  They are buffered until the
                        # route is known to be usable, then emitted once.
                        return _prepend(prelude, _empty_generator()), index
                    if isinstance(chunk, BaseException):
                        raise chunk
                    prelude.append(chunk)
                    if _is_visible_chunk(chunk):
                        return _prepend(prelude, iterator), index
            except Exception as exception:
                error = classify_exception(
                    exception,
                    provider=route.provider,
                    model=route.model,
                )
                if error.kind not in self.fallback_on or index == len(self.routes) - 1:
                    raise error from exception
                last_error = error
        raise last_error or VvLlmError(
            "no fallback route was eligible",
            kind=ErrorKind.CONFIGURATION,
        )


class AsyncFallbackChatClient:
    def __init__(
        self,
        registry: ProviderRegistry,
        routes: list[FallbackRoute] | tuple[FallbackRoute, ...],
        *,
        fallback_on: frozenset[ErrorKind] | None = None,
    ) -> None:
        if not routes:
            raise ValueError("fallback routes cannot be empty")
        self.registry = registry
        self.routes = tuple(routes)
        self.fallback_on = fallback_on or _default_fallback_errors()

    async def create(self, request: ChatRequest) -> Any:
        if request.stream:
            stream, _ = await self._prepare_stream_async(request)
            return stream
        response, _, _ = await self._complete_async(request)
        return response

    async def create_with_metadata(self, request: ChatRequest) -> CompletionResult[Any]:
        started = perf_counter()
        if request.stream:
            response, index = await self._prepare_stream_async(request)
            route = self.routes[index]
        else:
            response, index, route = await self._complete_async(request)
        return CompletionResult(
            response=response,
            metadata=_metadata(response, route, index, started),
        )

    async def _complete_async(self, request: ChatRequest) -> tuple[Any, int, FallbackRoute]:
        last_error: VvLlmError | None = None
        for index, route in enumerate(self.routes):
            registration = self.registry.get(route.provider)
            routed = request.model_copy(update={"model": route.model})
            capability_error = _capability_error(registration, routed)
            if capability_error is not None:
                last_error = capability_error
                continue
            try:
                return await registration.factory().create(routed), index, route
            except Exception as exception:
                error = classify_exception(
                    exception,
                    provider=route.provider,
                    model=route.model,
                )
                if error.kind not in self.fallback_on or index == len(self.routes) - 1:
                    raise error from exception
                last_error = error
        raise last_error or VvLlmError(
            "no fallback route was eligible",
            kind=ErrorKind.CONFIGURATION,
        )

    async def _prepare_stream_async(
        self,
        request: ChatRequest,
    ) -> tuple[AsyncGenerator[Any, None], int]:
        last_error: VvLlmError | None = None
        for index, route in enumerate(self.routes):
            registration = self.registry.get(route.provider)
            routed = request.model_copy(update={"model": route.model, "stream": True})
            capability_error = _capability_error(registration, routed)
            if capability_error is not None:
                last_error = capability_error
                continue
            try:
                iterator = await registration.factory().create(routed)
                prelude: list[Any] = []
                while True:
                    try:
                        chunk = await anext(iterator)
                    except StopAsyncIteration:
                        # Match the synchronous path and preserve a valid
                        # usage/done-only stream after route establishment.
                        return _prepend_async(prelude, _empty_async_generator()), index
                    if isinstance(chunk, BaseException):
                        raise chunk
                    prelude.append(chunk)
                    if _is_visible_chunk(chunk):
                        return _prepend_async(prelude, iterator), index
            except Exception as exception:
                error = classify_exception(
                    exception,
                    provider=route.provider,
                    model=route.model,
                )
                if error.kind not in self.fallback_on or index == len(self.routes) - 1:
                    raise error from exception
                last_error = error
        raise last_error or VvLlmError(
            "no fallback route was eligible",
            kind=ErrorKind.CONFIGURATION,
        )


def _capability_error(
    registration: ProviderRegistration,
    request: ChatRequest,
) -> VvLlmError | None:
    try:
        request.validate_capabilities(
            registration.capabilities,
            CapabilityPolicy.STRICT,
        )
    except ValueError as exception:
        return VvLlmError(
            str(exception),
            kind=ErrorKind.CONFIGURATION,
            provider=registration.name,
            model=request.model,
            source=exception,
        )
    return None


def _metadata(
    response: Any,
    route: FallbackRoute,
    index: int,
    started: float,
) -> ResponseMetadata:
    return ResponseMetadata(
        provider=route.provider,
        model=route.model,
        response_id=getattr(response, "id", None),
        request_id=getattr(response, "request_id", None),
        finish_reason=getattr(response, "finish_reason", None),
        latency_ms=(perf_counter() - started) * 1000,
        fallback_index=index,
    )


def _default_fallback_errors() -> frozenset[ErrorKind]:
    return frozenset(
        {
            ErrorKind.RATE_LIMITED,
            ErrorKind.NETWORK,
            ErrorKind.TIMEOUT,
            ErrorKind.PROVIDER_INTERNAL,
            ErrorKind.MODEL_NOT_FOUND,
        }
    )


def _is_visible_chunk(chunk: Any) -> bool:
    """Return whether a stream delta contains user-visible output.

    Provider adapters may emit usage-only, done, or empty deltas before the
    first content delta.  Those are transport metadata, not evidence that a
    route has established a visible response.  Keep this deliberately
    provider-neutral and accept both normalized objects and mapping-shaped
    test/provider payloads.
    """

    if isinstance(chunk, (str, bytes)):
        return bool(chunk)
    if isinstance(chunk, Mapping):
        return bool(chunk.get("content") or chunk.get("reasoning_content") or chunk.get("tool_calls"))
    return bool(getattr(chunk, "content", None) or getattr(chunk, "reasoning_content", None) or getattr(chunk, "tool_calls", None))


def _prepend(prelude: list[Any], iterator: Any) -> Generator[Any, None, None]:
    yield from prelude
    yield from iterator


def _empty_generator() -> Generator[Any, None, None]:
    if False:
        yield None


async def _prepend_async(prelude: list[Any], iterator: Any) -> AsyncGenerator[Any, None]:
    for chunk in prelude:
        yield chunk
    async for item in iterator:
        yield item


async def _empty_async_generator() -> AsyncGenerator[Any, None]:
    if False:
        yield None


__all__ = [
    "AsyncFallbackChatClient",
    "FallbackChatClient",
    "FallbackRoute",
    "ProviderRegistration",
    "ProviderRegistry",
]
