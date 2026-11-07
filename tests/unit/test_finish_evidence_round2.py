"""Review counterexamples over real SDKs and loopback HTTP."""

import asyncio
import gc
import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from types import SimpleNamespace

import httpx2
import openai._streaming as sdk
import pytest
from openai import AsyncStream, Stream

from unit import test_finish_evidence as f
from unit.test_finish_evidence import loopback_only  # noqa: F401
from vv_llm import ChatRequest, ChatRequestOptions, ModelCapabilities
from vv_llm.chat_clients import BackendType, create_async_chat_client, create_chat_client
from vv_llm.chat_clients import finish_evidence as evidence
from vv_llm.registry import AsyncFallbackChatClient, FallbackChatClient, FallbackRoute, ProviderRegistry


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("mode", ["missing", "signature"])
@pytest.mark.parametrize("case", ["done_only", "stop"])
def test_decoder_interface_drift(monkeypatch, caplog, is_async, mode, case):
    cls = AsyncStream if is_async else Stream
    init = cls.__init__
    original = sdk.SSEDecoder
    decoders = []

    class DecoderV2:
        def __init__(self):
            self.inner = original()

        def iter_bytes(self, *, source):
            yield from self.inner.iter_bytes(source)

        async def aiter_bytes(self, *, source):
            async for event in self.inner.aiter_bytes(source):
                yield event

    def new_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        if mode == "missing":
            self._decoder_v2 = self._decoder
            del self._decoder
        else:
            self._decoder = DecoderV2()
            decoders.append(self._decoder)

    def events(self):
        if mode == "missing":
            yield from self._decoder_v2.iter_bytes(self.response.iter_bytes())
        else:
            assert self._decoder is decoders[-1]
            yield from self._decoder.iter_bytes(source=self.response.iter_bytes())

    async def aevents(self):
        if mode == "missing":
            async for event in self._decoder_v2.aiter_bytes(self.response.aiter_bytes()):
                yield event
        else:
            assert self._decoder is decoders[-1]
            async for event in self._decoder.aiter_bytes(source=self.response.aiter_bytes()):
                yield event

    monkeypatch.setattr(cls, "__init__", new_init)
    monkeypatch.setattr(cls, "_iter_events", aevents if is_async else events)
    result, chunks = f.run_case("openai", case, True, is_async)
    assert len(chunks) == (1 if case == "done_only" else 2)
    # Previously False guarded against inventing DONE; None keeps that guard
    # while distinguishing reduced observation from a fully observed EOF.
    assert result.metadata.stream_terminated is (True if case == "stop" else None)
    assert result.metadata.termination_signals == (["finish_reason"] if case == "stop" else [])
    assert result.metadata.attributes["openai_done_observer"] == "unsupported_decoder"
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


@pytest.mark.parametrize("is_async", [False, True])
def test_observer_failure_does_not_break_sdk(monkeypatch, caplog, is_async):
    def broken(*args):
        raise ValueError("synthetic observer error")

    monkeypatch.setattr(evidence, "record_termination", broken)
    result, chunks = f.run_case("openai", "done_only", True, is_async)
    assert len(chunks) == 1
    assert result.metadata.stream_exhausted is True
    # Keep failure visible via the diagnostic/warning, without claiming no DONE.
    assert result.metadata.stream_terminated is None
    assert result.metadata.attributes["openai_done_observer"] == "observation_failed"
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


@pytest.mark.parametrize("is_async", [False, True])
def test_done_decoder_forwards_arguments_and_sdk_errors(is_async):
    token = object()
    error = ValueError("SDK parse failure")
    calls = []

    class Decoder:
        def iter_bytes(self, *args, **kwargs):
            calls.append((args, kwargs))
            yield SimpleNamespace(data="[DONE]")
            raise error

        async def aiter_bytes(self, *args, **kwargs):
            calls.append((args, kwargs))
            yield SimpleNamespace(data="[DONE]")
            raise error

    metadata = evidence.stream_metadata("openai", "test")
    decoder = evidence._DoneDecoder(Decoder(), metadata)

    async def run():
        iterator = decoder.aiter_bytes(token, source=token) if is_async else decoder.iter_bytes(token, source=token)
        assert (await anext(iterator) if is_async else next(iterator)).data == "[DONE]"
        with pytest.raises(ValueError) as caught:
            await anext(iterator) if is_async else next(iterator)
        assert caught.value is error

    asyncio.run(run())
    assert calls == [((token,), {"source": token})]
    assert metadata.termination_signals == ["[DONE]"]


def install_choices(monkeypatch, choices_by_frame):
    def wire(protocol, case, stream):
        base = {"id": "review", "created": 0, "model": "test-model"}
        if not stream:
            choices = [{**c, "message": {"role": "assistant", "content": f"body-{i}", "refusal": c["delta"].get("refusal")}} for i, c in enumerate(choices_by_frame[0])]
            return "application/json", json.dumps({**base, "object": "chat.completion", "choices": choices}).encode()
        return "text/event-stream", "".join(
            "data: " + json.dumps({**base, "object": "chat.completion.chunk", "choices": choices}) + "\n\n" for choices in choices_by_frame
        ).encode()

    monkeypatch.setattr(f, "wire", wire)


def choice(index, reason=None, refusal=None):
    return {"index": index, "delta": {"refusal": refusal}, "finish_reason": reason}


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "frames,reason,refusal",
    [
        ([[choice(0)], [choice(1, "stop")]], None, None),
        ([[choice(0)], [choice(1), choice(0, "length")]], "length", None),
        ([[choice(1, "length"), choice(0, "stop")]], "stop", None),
        ([[choice(0, refusal="No ")], [choice(1, "stop", "OTHER")], [choice(0, refusal="thanks")]], None, "No thanks"),
        ([[{"delta": {"refusal": "legacy"}, "finish_reason": "stop"}]], "stop", "legacy"),
    ],
)
def test_choice_zero_evidence(monkeypatch, is_async, frames, reason, refusal):
    install_choices(monkeypatch, frames)
    result, chunks = f.run_case("openai", "review", True, is_async)
    assert result.metadata.finish_reason == reason
    assert result.metadata.refusal == refusal
    assert result.metadata.stream_terminated is (reason is not None)
    assert len(chunks) == len(frames)  # Legacy content selection/sequence stays unchanged.


@pytest.mark.parametrize("is_async", [False, True])
def test_nonstream_choice_zero_evidence(monkeypatch, is_async):
    install_choices(monkeypatch, [[choice(1, "length", "OTHER"), choice(0, "stop", "No")]])
    result, _ = f.run_case("openai", "review", False, is_async)
    assert result.content == "body-0"
    assert result.metadata.finish_reason == "stop"
    assert result.metadata.refusal == "No"


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reason,normalized", [("model_context_window_exceeded", "length"), ("pause_turn", "other")])
def test_anthropic_context_and_pause(reason, normalized, stream, is_async):
    result, _ = f.run_case("anthropic", reason, stream, is_async)
    assert result.metadata.finish_reason == reason
    assert result.metadata.finish_reason_normalized == normalized


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("consume", ["none", "partial", "full"])
def test_fallback_buffered_exhaustion(monkeypatch, is_async, consume):
    original = f.wire

    def wire(protocol, case, stream):
        kind, body = original("openai", "empty_done", True)
        chunk = {"id": "review", "created": 0, "model": "test-model", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": None}]}
        usage = {**chunk, "choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}
        return kind, ("data: " + json.dumps(chunk) + "\n\ndata: " + json.dumps(usage) + "\n\n").encode() + body

    monkeypatch.setattr(f, "wire", wire)
    with f.server("openai", "review", True) as (base, requests):

        async def run():
            custom = httpx2.AsyncClient(trust_env=False) if is_async else httpx2.Client(trust_env=False)
            factory = create_async_chat_client if is_async else create_chat_client
            provider = factory(BackendType.OpenAI, model="test-model", settings=f.settings("openai", base, "openai"), http_client=custom)
            registry = ProviderRegistry()
            registry.register("openai", lambda: provider, capabilities=ModelCapabilities())
            client = (AsyncFallbackChatClient if is_async else FallbackChatClient)(registry, [FallbackRoute("openai", "test-model")])
            request = ChatRequest(messages=[{"role": "user", "content": "hello"}], model="test-model", stream=True, skip_cutoff=True, options=ChatRequestOptions(max_tokens=32))
            try:
                result = await client.create(request) if is_async else client.create(request)
                assert result.metadata.stream_terminated is True
                assert result.metadata.stream_exhausted is False
                if consume == "full":
                    chunks = [c async for c in result] if is_async else list(result)
                    assert len(chunks) == 2
                    assert result.metadata.stream_exhausted is True
                else:
                    if consume == "partial":
                        await anext(result) if is_async else next(result)
                    await result.aclose() if is_async else result.close()
                    assert result.metadata.stream_exhausted is False
                    with pytest.raises(StopAsyncIteration if is_async else StopIteration):
                        await anext(result) if is_async else next(result)
                    assert result.metadata.stream_exhausted is False
                assert result.metadata.termination_signals == ["[DONE]"]
                assert len(requests) == 1
            finally:
                await custom.aclose() if is_async else custom.close()

        asyncio.run(run())


@contextmanager
def held_server(protocol, retain_terminal=False):
    release = Event()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(self.rfile.read(int(self.headers["Content-Length"])))
            kind, body = f.wire(protocol, "completed" if protocol == "responses" else "end_turn" if protocol == "anthropic" else "stop", True)
            if retain_terminal:
                pass
            elif protocol == "openai":
                body = body.split(b"data: [DONE]")[0]
            elif protocol == "anthropic":
                body = body.split(b"event: message_delta")[0]
            else:
                body = body.split(b"event: response.completed")[0]
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body) + 20))
            self.end_headers()
            try:
                self.wfile.write(body)
                self.wfile.flush()
                release.wait(5)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        release.set()
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("protocol", ["openai", "anthropic", "responses", "anthropic_compat", "fallback"])
@pytest.mark.parametrize("end", ["close", "throw", "gc", "cancel"])
def test_async_http_response_is_closed(protocol, end):
    # Keep the response alive: closing the shared client must not count as cleanup.
    responses = []

    async def capture(response):
        responses.append(response)

    wire_protocol = "openai" if protocol in {"anthropic_compat", "fallback"} else protocol
    with held_server(wire_protocol) as (base, requests):

        async def run():
            async with httpx2.AsyncClient(trust_env=False, event_hooks={"response": [capture]}) as custom:
                backend = "anthropic" if protocol in {"anthropic", "anthropic_compat"} else "openai"
                client = create_async_chat_client(BackendType(backend), model="test-model", settings=f.settings(backend, base, wire_protocol), http_client=custom)
                if protocol == "fallback":
                    registry = ProviderRegistry()
                    registry.register("openai", lambda: client, capabilities=ModelCapabilities())
                    fallback = AsyncFallbackChatClient(registry, [FallbackRoute("openai", "test-model")])
                    stream = await fallback.create(
                        ChatRequest(messages=[{"role": "user", "content": "hello"}], model="test-model", stream=True, skip_cutoff=True, options=ChatRequestOptions(max_tokens=32))
                    )
                else:
                    stream = await client.create_completion(messages=[{"role": "user", "content": "hello"}], stream=True, max_tokens=32, skip_cutoff=True)
                await anext(stream)
                assert not responses[0].is_closed
                if end == "close":
                    await stream.aclose()
                elif end == "throw":
                    with pytest.raises(ValueError, match="probe"):
                        await stream.athrow(ValueError("probe"))
                elif end == "cancel":
                    # Cancel while the SDK waits on real HTTP, after available deltas.
                    async def drain():
                        async for _ in stream:
                            pass

                    task = asyncio.create_task(drain())
                    await asyncio.sleep(0.03)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                del stream
                gc.collect()
                for _ in range(100):
                    if responses[0].is_closed:
                        break
                    await asyncio.sleep(0.005)
                assert responses[0].is_closed
                assert len(requests) == 1

        asyncio.run(run())


@pytest.mark.parametrize("is_async", [False, True])
def test_sdk_json_parse_error_still_propagates(monkeypatch, is_async):
    original = f.wire

    def wire(protocol, case, stream):
        kind, body = original("openai", "eof", True)
        return kind, body + b"data: {malformed JSON\n\n"

    monkeypatch.setattr(f, "wire", wire)
    stream, chunks, error = f.run_case("openai", "review", True, is_async, capture_error=True)
    assert len(chunks) == 1
    assert error["class"] == "JSONDecodeError"
    assert stream.metadata.stream_terminated is False
    assert stream.metadata.stream_exhausted is False


def test_responses_cancel_after_usage_keeps_cancelled_error():
    responses = []

    async def capture(response):
        responses.append(response)

    with held_server("responses", retain_terminal=True) as (base, requests):

        async def run():
            async with httpx2.AsyncClient(trust_env=False, event_hooks={"response": [capture]}) as custom:
                client = create_async_chat_client(BackendType.OpenAI, model="test-model", settings=f.settings("openai", base, "responses"), http_client=custom)
                stream = await client.create_completion(messages=[{"role": "user", "content": "hello"}], stream=True, max_tokens=32, skip_cutoff=True)
                await anext(stream)

                async def drain():
                    async for _ in stream:
                        pass

                task = asyncio.create_task(drain())
                for _ in range(100):
                    if stream.metadata.stream_terminated:
                        break
                    await asyncio.sleep(0.005)
                assert stream.metadata.stream_terminated is True
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert responses[0].is_closed
                assert stream.metadata.stream_exhausted is False
                assert len(requests) == 1

        asyncio.run(run())
