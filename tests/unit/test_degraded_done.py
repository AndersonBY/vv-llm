"""Real OpenAI SSE parsing over MockTransport, including wrapper/replay paths."""

import asyncio
from types import SimpleNamespace

import httpx2
import pytest
from openai import AsyncStream, Stream
from openai.resources.chat.completions import AsyncCompletions, Completions

from unit import test_finish_evidence as f
from unit.test_finish_evidence import loopback_only  # noqa: F401
from vv_llm import (
    AsyncMiddlewareChatClient,
    ChatRequest,
    ChatRequestOptions,
    MiddlewareChatClient,
    ModelCapabilities,
)
from vv_llm.chat_clients import BackendType, create_async_chat_client, create_chat_client
from vv_llm.chat_clients import finish_evidence as evidence
from vv_llm.registry import AsyncFallbackChatClient, FallbackChatClient, FallbackRoute, ProviderRegistry


def degrade_observer(monkeypatch, mode, is_async):
    if mode == "unsupported_decoder":
        cls = AsyncStream if is_async else Stream
        original_init = cls.__init__

        class DecoderV2:
            def __init__(self, inner):
                self.inner = inner

            # The SDK can still call positionally; the observer rejects the rename.
            def iter_bytes(self, source):
                yield from self.inner.iter_bytes(source)

            async def aiter_bytes(self, source):
                async for event in self.inner.aiter_bytes(source):
                    yield event

        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._decoder = DecoderV2(self._decoder)

        monkeypatch.setattr(cls, "__init__", init)
    elif mode == "observation_failed":
        original_record = evidence.record_termination

        def record(metadata, signal):
            if signal == "[DONE]":
                raise ValueError("synthetic observer error")
            original_record(metadata, signal)

        monkeypatch.setattr(evidence, "record_termination", record)
    elif mode == "unsupported_stream":
        cls = AsyncCompletions if is_async else Completions
        original_create = cls.create

        class StreamV2:
            def __init__(self, inner):
                self.inner = inner

            def __iter__(self):
                return iter(self.inner)

            def __aiter__(self):
                return self.inner.__aiter__()

            def close(self):
                return self.inner.close()

        def create(self, *args, **kwargs):
            return StreamV2(original_create(self, *args, **kwargs))

        async def acreate(self, *args, **kwargs):
            return StreamV2(await original_create(self, *args, **kwargs))

        monkeypatch.setattr(cls, "create", acreate if is_async else create)


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("mode", [None, "unsupported_decoder", "observation_failed", "unsupported_stream"])
@pytest.mark.parametrize("case", ["done_only", "stop", "eof", "empty_done", "buffered_done"])
@pytest.mark.parametrize("wrapper", ["direct", "middleware", "fallback"])
def test_degraded_done_end_to_end(monkeypatch, caplog, is_async, mode, case, wrapper):
    degrade_observer(monkeypatch, mode, is_async)
    kind, body = f.wire("openai", "done_only" if case == "buffered_done" else case, True)
    if case == "buffered_done":
        body = body.replace(b'"answer"', b'""')
    requests = []

    def respond(request):
        f.check_request(request)
        requests.append(request)
        return httpx2.Response(200, headers={"content-type": kind}, content=body)

    async def run():
        custom = (httpx2.AsyncClient if is_async else httpx2.Client)(transport=httpx2.MockTransport(respond), trust_env=False)
        factory = create_async_chat_client if is_async else create_chat_client
        client = factory(BackendType.OpenAI, model="test-model", settings=f.settings("openai", "http://127.0.0.1/v1", "openai"), http_client=custom)
        if "fallback" in wrapper:
            provider = client
            # Bind the provider before replacing client with its fallback wrapper.
            registry = ProviderRegistry()
            registry.register("openai", lambda: provider, capabilities=ModelCapabilities())
            client = (AsyncFallbackChatClient if is_async else FallbackChatClient)(registry, [FallbackRoute("openai", "test-model")])
        if "middleware" in wrapper:
            client = (AsyncMiddlewareChatClient if is_async else MiddlewareChatClient)(client)
        request = ChatRequest(messages=[{"role": "user", "content": "hello"}], model="test-model", stream=True, skip_cutoff=True, options=ChatRequestOptions(max_tokens=32))
        try:
            if wrapper == "direct":
                stream = await client.create(request) if is_async else client.create(request)
                metadata = stream.metadata
            else:
                result = await client.create_with_metadata(request) if is_async else client.create_with_metadata(request)
                stream, metadata = result.response, result.metadata
                assert metadata is stream.metadata
            assert metadata.stream_exhausted is False
            # Interface degradation is immediate; observation_failed occurs at DONE.
            if mode in {"unsupported_decoder", "unsupported_stream"}:
                assert metadata.stream_terminated is None
            chunks = [c async for c in stream] if is_async else list(stream)
            assert "".join(c.content or "" for c in chunks) == ("" if case in {"empty_done", "buffered_done"} else "answer")
            assert len(chunks) == (0 if case == "empty_done" else 2 if case == "stop" else 1)
            assert all(c.metadata is metadata for c in chunks)
            assert metadata.stream_exhausted is True
            degraded = mode is not None and (mode != "observation_failed" or case != "eof")
            expected = True if case == "stop" or (mode is None and case != "eof") else None if degraded else False
            assert metadata.stream_terminated is expected
            signals = (["finish_reason"] if case == "stop" else []) + (["[DONE]"] if mode is None and case != "eof" else [])
            assert metadata.termination_signals == signals
            if degraded:
                assert metadata.attributes["openai_done_observer"] == mode
                warnings = [r.getMessage() for r in caplog.records if r.name == evidence.__name__ and r.levelname == "WARNING"]
                assert warnings == [f"OpenAI [DONE] observer disabled ({mode}); retaining other observable termination signals"]
            else:
                assert "openai_done_observer" not in metadata.attributes
            assert len(requests) == 1
        finally:
            await custom.aclose() if is_async else custom.close()

    asyncio.run(run())


@pytest.mark.parametrize("signal", ["finish_reason", "message_stop", "response.completed"])
@pytest.mark.parametrize("before", [False, True])
def test_other_terminal_signals_survive_degradation(monkeypatch, signal, before):
    degrade_observer(monkeypatch, "observation_failed", False)
    metadata = evidence.stream_metadata("test", "test-model")

    def terminate():
        if signal == "finish_reason":
            evidence.observe_chat_chunk(metadata, SimpleNamespace(choices=[SimpleNamespace(index=0, finish_reason="stop", delta=SimpleNamespace(refusal=None))]))
        elif signal == "message_stop":
            evidence.observe_anthropic_event(metadata, SimpleNamespace(type=signal))
        else:
            evidence.observe_response_event(metadata, SimpleNamespace(type=signal, response=None), {})

    if before:
        terminate()
    decoder = evidence._DoneDecoder(None, metadata)
    event = SimpleNamespace(data="[DONE]")
    assert decoder._observe(event) is event
    assert metadata.stream_terminated is (True if before else None)
    if not before:
        terminate()
    assert metadata.stream_terminated is True
    assert metadata.termination_signals == [signal]
    assert metadata.attributes["openai_done_observer"] == "observation_failed"
