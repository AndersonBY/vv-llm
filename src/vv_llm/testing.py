from __future__ import annotations

from collections import deque
from collections.abc import AsyncGenerator, Generator, Iterable
from dataclasses import dataclass
from threading import Lock
from types import SimpleNamespace
from typing import Any

from .middleware import _request_from_completion_kwargs
from .types.chat_request import ChatRequest, ModelCapabilities


@dataclass(frozen=True)
class ScriptedStream:
    chunks: tuple[Any, ...]

    def __init__(self, chunks: Iterable[Any]) -> None:
        object.__setattr__(self, "chunks", tuple(chunks))


class ScriptedChatClient:
    def __init__(
        self,
        steps: Iterable[Any | BaseException],
        *,
        provider: str = "scripted",
        model: str = "scripted-model",
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        self.backend_name = SimpleNamespace(value=provider)
        self.model = model
        self.capabilities = capabilities or ModelCapabilities()
        self._steps = deque(steps)
        self._requests: list[ChatRequest] = []
        self._lock = Lock()

    @property
    def requests(self) -> tuple[ChatRequest, ...]:
        with self._lock:
            return tuple(request.model_copy(deep=True) for request in self._requests)

    def create(self, request: ChatRequest, **_: Any) -> Any:
        with self._lock:
            self._requests.append(request.model_copy(deep=True))
            if not self._steps:
                raise AssertionError("scripted client has no remaining steps")
            step = self._steps.popleft()
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, ScriptedStream):
            return self._stream(step)
        return step

    def create_completion(self, **kwargs: Any) -> Any:
        return self.create(_request_from_completion_kwargs(kwargs))

    def create_stream(self, **kwargs: Any) -> Generator[Any, None, None]:
        return self.create_completion(**kwargs, stream=True)

    @staticmethod
    def _stream(step: ScriptedStream) -> Generator[Any, None, None]:
        for chunk in step.chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk


class AsyncScriptedChatClient:
    def __init__(
        self,
        steps: Iterable[Any | BaseException],
        *,
        provider: str = "scripted",
        model: str = "scripted-model",
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        self.backend_name = SimpleNamespace(value=provider)
        self.model = model
        self.capabilities = capabilities or ModelCapabilities()
        self._steps = deque(steps)
        self._requests: list[ChatRequest] = []
        self._lock = Lock()

    @property
    def requests(self) -> tuple[ChatRequest, ...]:
        with self._lock:
            return tuple(request.model_copy(deep=True) for request in self._requests)

    async def create(self, request: ChatRequest, **_: Any) -> Any:
        with self._lock:
            self._requests.append(request.model_copy(deep=True))
            if not self._steps:
                raise AssertionError("scripted client has no remaining steps")
            step = self._steps.popleft()
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, ScriptedStream):
            return self._async_stream(step)
        return step

    async def create_completion(self, **kwargs: Any) -> Any:
        return await self.create(_request_from_completion_kwargs(kwargs))

    async def create_stream(self, **kwargs: Any) -> AsyncGenerator[Any, None]:
        return await self.create_completion(**kwargs, stream=True)

    @staticmethod
    async def _async_stream(step: ScriptedStream) -> AsyncGenerator[Any, None]:
        for chunk in step.chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk


__all__ = [
    "AsyncScriptedChatClient",
    "ScriptedChatClient",
    "ScriptedStream",
]
