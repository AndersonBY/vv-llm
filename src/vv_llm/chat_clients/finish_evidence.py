"""Observe provider evidence without changing emitted deltas or error handling."""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncGenerator, Generator
from typing import Any

from openai import AsyncStream, Stream

from ..types.llm_parameters import ChatCompletionDeltaMessage
from ..types.response import FinishReason, ResponseMetadata


_LOGGER = logging.getLogger(__name__)


_REASONS: dict[str, FinishReason] = {
    "stop": "stop",
    "end_turn": "stop",
    "stop_sequence": "stop",
    "completed": "stop",
    "length": "length",
    "max_tokens": "length",
    "max_output_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "tool_use": "tool_calls",
    "content_filter": "content_filter",
    "refusal": "refusal",
}


def set_finish_reason(metadata: ResponseMetadata, reason: str | None) -> None:
    if reason is not None:
        metadata.finish_reason = reason
        metadata.finish_reason_normalized = _REASONS.get(reason, "other")


def record_termination(metadata: ResponseMetadata, signal: str) -> None:
    metadata.stream_terminated = True
    if metadata.termination_signals is None:
        metadata.termination_signals = []
    if signal not in metadata.termination_signals:
        metadata.termination_signals.append(signal)


def stream_metadata(provider: str, model: str) -> ResponseMetadata:
    return ResponseMetadata(provider=provider, model=model, stream_terminated=False, stream_exhausted=False, termination_signals=[])


def _choice_zero(choices: Any) -> Any:
    for choice in choices:
        if getattr(choice, "index", None) == 0:
            return choice
    # Some compatible providers omit indices entirely.
    if choices and all(getattr(choice, "index", None) is None for choice in choices):
        return choices[0]
    return None


def observe_chat_response(metadata: ResponseMetadata, response: Any) -> None:
    choice = _choice_zero(response.choices)
    if choice is not None:
        set_finish_reason(metadata, getattr(choice, "finish_reason", None))
        metadata.refusal = getattr(choice.message, "refusal", None)


def observe_chat_chunk(metadata: ResponseMetadata, chunk: Any) -> None:
    choice = _choice_zero(chunk.choices)
    if choice is not None:
        reason = getattr(choice, "finish_reason", None)
        if reason is not None:
            set_finish_reason(metadata, reason)
            record_termination(metadata, "finish_reason")
        refusal = getattr(choice.delta, "refusal", None)
        if refusal is not None:
            metadata.refusal = (metadata.refusal or "") + refusal


def copy_evidence(target: ResponseMetadata, source: ResponseMetadata) -> None:
    for name in ("finish_reason", "finish_reason_normalized", "refusal", "stream_terminated", "stream_exhausted", "termination_signals"):
        setattr(target, name, getattr(source, name))
    target.attributes.update(source.attributes)


def observe_anthropic_event(metadata: ResponseMetadata, event: Any) -> None:
    if event.type == "message_delta":
        reason = getattr(event.delta, "stop_reason", None)
        if reason is not None:
            set_finish_reason(metadata, reason)
            record_termination(metadata, "message_delta.stop_reason")
    elif event.type == "message_stop":
        record_termination(metadata, "message_stop")


def observe_response(metadata: ResponseMetadata, response: Any) -> None:
    if response is None:
        return
    status = getattr(response, "status", None)
    details = getattr(response, "incomplete_details", None)
    reason = getattr(details, "reason", None)
    set_finish_reason(metadata, reason if reason is not None else status)
    for name in ("status", "incomplete_details", "error"):
        value = getattr(response, name, None)
        if value is not None:
            metadata.attributes[name] = value.model_dump() if hasattr(value, "model_dump") else value
    refusals = [part.refusal for item in (getattr(response, "output", None) or []) for part in (getattr(item, "content", None) or []) if getattr(part, "type", None) == "refusal"]
    if refusals:
        metadata.refusal = "".join(refusals)


def observe_response_event(metadata: ResponseMetadata, event: Any, refusals: dict[tuple[int, int], str]) -> None:
    if event.type in {"response.completed", "response.incomplete", "response.failed"}:
        observe_response(metadata, event.response)
        record_termination(metadata, event.type)
    elif event.type in {"response.refusal.delta", "response.refusal.done"}:
        key = (event.output_index, event.content_index)
        refusals[key] = refusals.get(key, "") + event.delta if event.type == "response.refusal.delta" else event.refusal
        metadata.refusal = "".join(refusals.values())


def _done_observer_unavailable(metadata: ResponseMetadata, reason: str) -> None:
    if metadata.stream_terminated is not True:
        metadata.stream_terminated = None
    if "openai_done_observer" not in metadata.attributes:
        metadata.attributes["openai_done_observer"] = reason
        _LOGGER.warning("OpenAI [DONE] observer disabled (%s); retaining other observable termination signals", reason)


class _DoneDecoder:
    """Tap the SDK's SSE decoder before it consumes and hides [DONE]."""

    def __init__(self, decoder: Any, metadata: ResponseMetadata):
        self.decoder = decoder
        self.metadata = metadata
        self._disabled = False

    def _observe(self, event: Any) -> Any:
        if not self._disabled:
            try:
                if event.data.startswith("[DONE]"):
                    record_termination(self.metadata, "[DONE]")
            except Exception:
                self._disabled = True
                _done_observer_unavailable(self.metadata, "observation_failed")
        return event

    def iter_bytes(self, *args: Any, **kwargs: Any):
        for event in self.decoder.iter_bytes(*args, **kwargs):
            yield self._observe(event)

    async def aiter_bytes(self, *args: Any, **kwargs: Any):
        async for event in self.decoder.aiter_bytes(*args, **kwargs):
            yield self._observe(event)


def observe_openai_done(stream: Any, metadata: ResponseMetadata) -> None:
    if not isinstance(stream, Stream | AsyncStream):
        _done_observer_unavailable(metadata, "unsupported_stream")
        return
    # Only install on the known private SDK interface, before consuming any bytes.
    try:
        decoder = stream._decoder
        for name in ("iter_bytes", "aiter_bytes"):
            parameters = tuple(inspect.signature(getattr(decoder, name)).parameters.values())
            if len(parameters) != 1 or parameters[0].name != "iterator" or parameters[0].kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                _done_observer_unavailable(metadata, "unsupported_decoder")
                return
        stream._decoder = _DoneDecoder(decoder, metadata)
    except Exception:
        _done_observer_unavailable(metadata, "unsupported_decoder")


class CompletionStream(Generator[ChatCompletionDeltaMessage, Any, None]):
    """Generator protocol plus per-call evidence, also available for empty streams."""

    def __init__(self, iterator: Generator[ChatCompletionDeltaMessage, Any, None], metadata: ResponseMetadata):
        self._iterator = iterator
        self.metadata = metadata
        self._interrupted = False

    def _copy_evidence(self) -> None:
        source = getattr(self._iterator, "metadata", None)
        if isinstance(source, ResponseMetadata) and source is not self.metadata:
            copy_evidence(self.metadata, source)

    def _attach(self, chunk: ChatCompletionDeltaMessage) -> ChatCompletionDeltaMessage:
        self._copy_evidence()
        chunk.metadata = self.metadata
        return chunk

    def send(self, value: Any) -> ChatCompletionDeltaMessage:
        try:
            return self._attach(self._iterator.send(value))
        except StopIteration:
            self._copy_evidence()
            if not self._interrupted:
                self.metadata.stream_exhausted = True
            raise
        except BaseException:
            self._interrupted = True
            self._copy_evidence()
            raise

    def throw(self, *args: Any) -> ChatCompletionDeltaMessage:
        try:
            return self._attach(self._iterator.throw(*args))
        except BaseException:
            self._interrupted = True
            self._copy_evidence()
            raise

    def close(self) -> None:
        self._interrupted = True
        self._iterator.close()


class AsyncCompletionStream(AsyncGenerator[ChatCompletionDeltaMessage, Any]):
    """Asynchronous counterpart; does not read ahead or add terminal chunks."""

    def __init__(self, iterator: AsyncGenerator[ChatCompletionDeltaMessage, Any], metadata: ResponseMetadata):
        self._iterator = iterator
        self.metadata = metadata
        self._interrupted = False

    def _copy_evidence(self) -> None:
        source = getattr(self._iterator, "metadata", None)
        if isinstance(source, ResponseMetadata) and source is not self.metadata:
            copy_evidence(self.metadata, source)

    async def asend(self, value: Any) -> ChatCompletionDeltaMessage:
        try:
            chunk = await self._iterator.asend(value)
        except StopAsyncIteration:
            self._copy_evidence()
            if not self._interrupted:
                self.metadata.stream_exhausted = True
            raise
        except BaseException:
            self._interrupted = True
            self._copy_evidence()
            raise
        self._copy_evidence()
        chunk.metadata = self.metadata
        return chunk

    async def athrow(self, *args: Any) -> ChatCompletionDeltaMessage:
        try:
            chunk = await self._iterator.athrow(*args)
        except BaseException:
            self._interrupted = True
            self._copy_evidence()
            raise
        self._copy_evidence()
        chunk.metadata = self.metadata
        return chunk

    async def aclose(self) -> None:
        self._interrupted = True
        await self._iterator.aclose()
