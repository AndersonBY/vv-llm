"""Real SDK conversions over loopback HTTP; no provider credentials or network."""

from __future__ import annotations

import asyncio
import base64
import struct
import zlib
import json
import socket
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import httpx2
import pytest

from vv_llm.chat_clients import BackendType, SYNC_BACKEND_MAP, create_async_chat_client, create_chat_client


@pytest.fixture(autouse=True)
def loopback_only(monkeypatch):
    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect

    def checked_getaddrinfo(host, *args, **kwargs):
        assert host in {"127.0.0.1", "localhost", "::1", b"127.0.0.1", b"localhost", b"::1"}, f"Blocked non-loopback DNS: {host}"
        return original_getaddrinfo(host, *args, **kwargs)

    def checked_connect(sock, address):
        if sock.family in {socket.AF_INET, socket.AF_INET6}:
            assert address[0] in {"127.0.0.1", "::1"}, f"Blocked non-loopback connection: {address[0]}"
        return original_connect(sock, address)

    monkeypatch.setattr(socket, "getaddrinfo", checked_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", checked_connect)


def check_request(request):
    assert request.url.host in {"127.0.0.1", "localhost", "::1"}, f"Blocked non-loopback HTTP: {request.url.host}"


async def check_async_request(request):
    check_request(request)


LEGACY = Path(__file__).with_name("fixtures") / "finish_evidence_legacy.json"
OPENAI_CASES = ["stop", "length", "tool_calls", "content_filter", "refusal", "done_only", "reason_only", "eof", "empty_done", "empty_eof", "unknown"]
ANTHROPIC_CASES = ["end_turn", "max_tokens", "tool_use", "refusal", "stop_only", "reason_only", "eof", "empty_stop", "empty_eof", "unknown"]
RESPONSES_CASES = ["completed", "max_output_tokens", "content_filter", "refusal", "failed", "eof", "empty_eof"]


def wire(protocol, case, stream):
    if protocol == "anthropic":
        reason = {"reason_only": "end_turn", "unknown": "future_reason"}.get(case, case)
        message = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "test-model",
            "content": [{"type": "text", "text": "answer"}],
            "stop_reason": reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 2, "output_tokens": 1},
        }
        if not stream:
            return "application/json", json.dumps(message).encode()
        events = [{"type": "message_start", "message": {**message, "content": [], "stop_reason": None}}]
        if not case.startswith("empty"):
            events += [
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "answer"}},
                {"type": "content_block_stop", "index": 0},
            ]
        if case not in {"eof", "stop_only", "empty_stop", "empty_eof"}:
            events += [{"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": None}, "usage": {"output_tokens": 1}}]
        if case not in {"eof", "reason_only", "empty_eof"}:
            events += [{"type": "message_stop"}]
        return "text/event-stream", "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
    if protocol == "responses":
        status = "incomplete" if case in {"max_output_tokens", "content_filter"} else "failed" if case == "failed" else "completed"
        output = [
            {
                "type": "message",
                "id": "m1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "refusal", "refusal": "Synthetic refusal"} if case == "refusal" else {"type": "output_text", "text": "answer", "annotations": []}],
            }
        ]
        response = {
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "model": "test-model",
            "status": status,
            "output": output,
            "incomplete_details": {"reason": case} if status == "incomplete" else None,
            "error": {"code": "server_error", "message": "synthetic"} if status == "failed" else None,
            "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
        }
        if not stream:
            return "application/json", json.dumps(response).encode()
        events = []
        if case != "empty_eof":
            events.append({"type": "response.created", "response": {**response, "status": "in_progress", "output": []}, "sequence_number": 0})
            events += [
                {"type": "response.output_item.added", "output_index": 0, "item": {**output[0], "content": []}, "sequence_number": 1},
                {
                    "type": "response.content_part.added",
                    "output_index": 0,
                    "item_id": "m1",
                    "content_index": 0,
                    "part": {"type": "refusal", "refusal": ""} if case == "refusal" else {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
                    "sequence_number": 2,
                },
            ]
            if case == "refusal":
                events += [
                    {"type": "response.refusal.delta", "delta": "Synthetic ", "item_id": "m1", "output_index": 0, "content_index": 0, "sequence_number": 1},
                    {"type": "response.refusal.delta", "delta": "refusal", "item_id": "m1", "output_index": 0, "content_index": 0, "sequence_number": 2},
                    {"type": "response.refusal.done", "refusal": "Synthetic refusal", "item_id": "m1", "output_index": 0, "content_index": 0, "sequence_number": 3},
                ]
            else:
                events.append(
                    {"type": "response.output_text.delta", "delta": "answer", "item_id": "m1", "output_index": 0, "content_index": 0, "sequence_number": 1, "logprobs": []}
                )
            if case != "eof":
                events.append({"type": f"response.{status}", "response": response, "sequence_number": 4})
        return "text/event-stream", "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
    reason = {"refusal": "stop", "unknown": "future_reason", "reason_only": "stop"}.get(case, case)
    message = {"content": "answer", "refusal": "Synthetic refusal" if case == "refusal" else None}
    base = {"id": "chat_test", "created": 0, "model": "test-model"}
    if not stream:
        return "application/json", json.dumps(
            {
                **base,
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": reason}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        ).encode()
    chunks = [] if case.startswith("empty") else [{**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": message, "finish_reason": None}]}]
    if case not in {"done_only", "eof", "empty_done", "empty_eof"}:
        chunks += [{**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    if case not in {"eof", "empty_eof", "reason_only"}:
        body += "data: [DONE]\n\n"
    return "text/event-stream", body.encode()


@contextmanager
def server(protocol, case, stream, transport=None):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert (self.path.endswith("invoke-with-response-stream") if transport == "anthropic_bedrock" else request.get("stream", False)) is stream
            requests.append(request)
            content_type, body = (
                wire(protocol, "completed" if protocol == "responses" else "stop" if protocol == "openai" else "end_turn", stream)
                if case == "error"
                else wire(protocol, case, stream)
            )
            if case == "error":
                # Drop terminal events, then fail after the first content chunk.
                if protocol == "openai":
                    body = body.split(b"\n\n")[0] + b"\n\n"
                    event = {"error": {"message": "synthetic stream error", "type": "server_error", "code": "server_error"}}
                elif protocol == "anthropic":
                    body = body.split(b"event: content_block_stop")[0]
                    event = {"type": "error", "error": {"type": "overloaded_error", "message": "synthetic stream error"}}
                else:
                    body = body.split(b"event: response.completed")[0]
                    event = {
                        "type": "error",
                        "code": "server_error",
                        "message": "synthetic stream error",
                        "param": None,
                        "sequence_number": 9,
                        "error": {"message": "synthetic stream error"},
                    }
                body += f"event: error\ndata: {json.dumps(event)}\n\n".encode()
            if transport == "anthropic_bedrock" and stream:
                frames = []
                for line in body.splitlines():
                    if not line.startswith(b"data: "):
                        continue
                    payload = json.dumps({"bytes": base64.b64encode(line[6:]).decode()}).encode()
                    headers = b""
                    for name, value in [(b":message-type", b"event"), (b":event-type", b"chunk"), (b":content-type", b"application/json")]:
                        headers += bytes([len(name)]) + name + b"\x07" + struct.pack(">H", len(value)) + value
                    prelude = struct.pack(">II", 16 + len(headers) + len(payload), len(headers))
                    frame = prelude + struct.pack(">I", zlib.crc32(prelude)) + headers + payload
                    frames.append(frame + struct.pack(">I", zlib.crc32(frame)))
                content_type, body = "application/vnd.amazon.eventstream", b"".join(frames)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=lambda: http.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}/v1", requests
    finally:
        http.shutdown()
        thread.join()
        http.server_close()


def settings(backend, base, protocol):
    return {
        "rate_limit": {"enabled": False},
        "endpoints": [
            {
                "id": "test",
                "api_base": base,
                "api_key": "synthetic-key",
                "endpoint_type": "anthropic" if protocol == "anthropic" else "openai",
                "response_api": protocol == "responses",
            }
        ],
        "backends": {
            backend: {"models": {"test-model": {"id": "test-model", "endpoints": ["test"], "context_length": 8192, "max_output_tokens": 32, "function_call_available": True}}}
        },
    }


def run_case(protocol, case, stream, is_async=False, backend=None, early=False, capture_error=False, transport=None):
    backend = backend or ("anthropic" if protocol == "anthropic" else "openai")
    with server(protocol, case, stream, transport=transport) as (base, requests):

        async def run():
            custom = (
                httpx2.AsyncClient(base_url=base, trust_env=False, event_hooks={"request": [check_async_request]})
                if is_async
                else httpx2.Client(base_url=base, trust_env=False, event_hooks={"request": [check_request]})
            )
            factory = create_async_chat_client if is_async else create_chat_client
            configured = settings(backend, base, protocol)
            if transport:
                configured["endpoints"][0].update(
                    {
                        "endpoint_type": transport,
                        "region": "us-east-1",
                        "credentials": {"quota_project_id": "synthetic", "access_key": "synthetic", "secret_key": "synthetic"},
                        "access_token": "synthetic-token",
                        "access_token_expires_at": 4102444800,
                    }
                )
            client = factory(BackendType(backend), model="test-model", settings=configured, http_client=custom)
            try:
                kwargs = {"messages": [{"role": "user", "content": "hello"}], "stream": stream, "max_tokens": 32, "skip_cutoff": True}
                result = await client.create_completion(**kwargs) if is_async else client.create_completion(**kwargs)
                chunks = []
                if stream:
                    if early:
                        chunks.append(await anext(result) if is_async else next(result))
                        if is_async:
                            await result.aclose()
                        else:
                            result.close()
                    else:
                        try:
                            if is_async:
                                async for chunk in result:
                                    chunks.append(chunk)
                            else:
                                for chunk in result:
                                    chunks.append(chunk)
                        except Exception as error:
                            if capture_error:
                                assert len(requests) == 1
                                return result, chunks, {"class": type(error).__name__, "message": str(error)}
                            raise
                else:
                    chunks = [result]
                assert len(requests) == 1
                return result, chunks
            finally:
                if is_async:
                    await custom.aclose()
                else:
                    custom.close()

        return asyncio.run(run())


def legacy_shape(chunks):
    return [chunk.model_dump(exclude={"metadata", "refusal"}) for chunk in chunks]


def all_cases():
    for protocol, cases in [("openai", OPENAI_CASES), ("anthropic", ANTHROPIC_CASES), ("responses", RESPONSES_CASES)]:
        for stream in (False, True):
            for case in cases:
                if not stream and case in {"done_only", "eof", "empty_done", "empty_eof", "stop_only", "reason_only", "empty_stop"}:
                    continue
                yield protocol, case, stream


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("protocol,case,stream", list(all_cases()))
def test_finish_evidence_and_legacy_sequence(protocol, case, stream, is_async):
    result, chunks = run_case(protocol, case, stream, is_async)
    baseline = json.loads(LEGACY.read_text())
    key = f"{protocol}/{case}/{stream}/{is_async}"
    assert legacy_shape(chunks) == baseline[key]
    metadata = result.metadata
    if chunks:
        assert all(chunk.metadata is metadata for chunk in chunks)
    expected = {"unknown": "future_reason", "reason_only": "end_turn" if protocol == "anthropic" else "stop", "refusal": "stop" if protocol == "openai" else "refusal"}.get(
        case, case
    )
    if protocol == "responses":
        expected = "completed" if case == "refusal" else case
    if case in {"done_only", "eof", "empty_done", "empty_eof", "stop_only", "empty_stop"}:
        expected = None
    assert metadata.finish_reason == expected
    normalized = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "future_reason": "other",
        "completed": "stop",
        "max_output_tokens": "length",
        "failed": "other",
    }.get(expected, expected)
    assert metadata.finish_reason_normalized == normalized
    assert metadata.refusal == ("Synthetic refusal" if case == "refusal" and protocol != "anthropic" else None)
    if stream:
        assert metadata.stream_exhausted is True
        assert metadata.stream_terminated is (case not in {"eof", "empty_eof"})
        if protocol == "openai" and case not in {"eof", "empty_eof", "reason_only"}:
            assert "[DONE]" in metadata.termination_signals
        if protocol == "anthropic":
            assert ("message_stop" in metadata.termination_signals) is (case not in {"eof", "empty_eof", "reason_only"})
    else:
        assert metadata.stream_terminated is None


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("backend", [backend.value for backend in SYNC_BACKEND_MAP])
def test_every_provider_compatible_route(backend, is_async):
    # Anthropic's configured OpenAI route is covered here as well as its native suite.
    result, chunks = run_case("openai", "content_filter", True, is_async, backend=backend)
    assert result.metadata.finish_reason == "content_filter"
    assert result.metadata.finish_reason_normalized == "content_filter"
    assert result.metadata.stream_terminated is True
    assert len(chunks) == 2


@pytest.mark.parametrize("protocol", ["openai", "anthropic", "responses"])
@pytest.mark.parametrize("is_async", [False, True])
def test_early_close_does_not_claim_exhaustion(protocol, is_async):
    result, _ = run_case(protocol, "completed" if protocol == "responses" else "stop", True, is_async, early=True)
    assert result.metadata.stream_exhausted is False
    assert result.metadata.stream_terminated is False


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("wrapper", ["middleware", "fallback"])
@pytest.mark.parametrize("case,stream", [("refusal", False), ("done_only", True), ("empty_done", True), ("empty_eof", True)])
def test_execution_wrappers_keep_live_evidence(is_async, wrapper, case, stream):
    from vv_llm import ChatRequest, ChatRequestOptions, ModelCapabilities
    from vv_llm.middleware import MiddlewareChatClient, AsyncMiddlewareChatClient
    from vv_llm.registry import ProviderRegistry, FallbackChatClient, AsyncFallbackChatClient, FallbackRoute

    with server("openai", case, stream) as (base, requests):

        async def run():
            custom = (
                httpx2.AsyncClient(base_url=base, trust_env=False, event_hooks={"request": [check_async_request]})
                if is_async
                else httpx2.Client(base_url=base, trust_env=False, event_hooks={"request": [check_request]})
            )
            factory = create_async_chat_client if is_async else create_chat_client
            provider = factory(BackendType.OpenAI, model="test-model", settings=settings("openai", base, "openai"), http_client=custom)
            if wrapper == "middleware":
                client = (AsyncMiddlewareChatClient if is_async else MiddlewareChatClient)(provider)
            else:
                registry = ProviderRegistry()
                registry.register("openai", lambda: provider, capabilities=ModelCapabilities())
                client = (AsyncFallbackChatClient if is_async else FallbackChatClient)(registry, [FallbackRoute("openai", "test-model")])
            try:
                request = ChatRequest(
                    messages=[{"role": "user", "content": "hello"}], model="test-model", stream=stream, skip_cutoff=True, options=ChatRequestOptions(max_tokens=32)
                )
                result = await client.create_with_metadata(request) if is_async else client.create_with_metadata(request)
                assert result.metadata is result.response.metadata
                if stream:
                    chunks = [chunk async for chunk in result.response] if is_async else list(result.response)
                    assert len(chunks) == (0 if case.startswith("empty") else 1)
                    assert result.metadata.stream_exhausted is True
                    assert result.metadata.stream_terminated is (case != "empty_eof")
                else:
                    assert result.metadata.refusal == "Synthetic refusal"
                    assert result.metadata.finish_reason == "stop"
                assert len(requests) == 1
            finally:
                if is_async:
                    await custom.aclose()
                else:
                    custom.close()

        asyncio.run(run())


@pytest.mark.parametrize("protocol", ["openai", "anthropic", "responses"])
@pytest.mark.parametrize("is_async", [False, True])
def test_stream_errors_preserve_legacy_exception_and_sequence(protocol, is_async):
    result, chunks, error = run_case(protocol, "error", True, is_async, capture_error=True)
    baseline = json.loads(LEGACY.with_name("finish_evidence_errors_legacy.json").read_text())
    assert {"chunks": legacy_shape(chunks), "error": error} == baseline[f"{protocol}/{is_async}"]
    assert result.metadata.stream_exhausted is False
    assert result.metadata.stream_terminated is False


@pytest.mark.parametrize("transport", ["openai_azure", "openai_vertex", "anthropic_vertex", "anthropic_bedrock"])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_transport_variants_keep_provider_evidence(transport, is_async, stream, monkeypatch):
    protocol = "anthropic" if transport.startswith("anthropic") else "openai"
    if transport == "anthropic_bedrock":
        pytest.importorskip("boto3")
        # Exercise real HTTP and AWS event decoding, without loading AWS credential files or signing.
        monkeypatch.setattr("anthropic.lib.bedrock._auth.get_auth_headers", lambda **kwargs: {})
        from anthropic import AsyncAnthropicBedrock

        class LoopbackAsyncBedrock(AsyncAnthropicBedrock):
            def __init__(self, **kwargs):
                # The existing async adapter omits api_base; pin the test SDK to loopback.
                kwargs["base_url"] = str(kwargs["http_client"].base_url)
                super().__init__(**kwargs)

        monkeypatch.setattr("vv_llm.chat_clients.anthropic_client.AsyncAnthropicBedrock", LoopbackAsyncBedrock)
    result, chunks = run_case(protocol, "refusal", stream, is_async, transport=transport)
    assert result.metadata.finish_reason == ("refusal" if protocol == "anthropic" else "stop")
    assert result.metadata.refusal == (None if protocol == "anthropic" else "Synthetic refusal")
    if stream:
        assert result.metadata.stream_exhausted is True
        assert result.metadata.stream_terminated is True
    assert len(chunks) == (3 if stream and protocol == "anthropic" else 2 if stream else 1)


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("end", ["close", "error"])
def test_repeated_next_after_interruption_does_not_claim_normal_exhaustion(is_async, end):
    from vv_llm.chat_clients.finish_evidence import CompletionStream, AsyncCompletionStream, stream_metadata
    from vv_llm.types import ChatCompletionDeltaMessage

    async def run():
        evidence = stream_metadata("openai", "test-model")

        def generate():
            yield ChatCompletionDeltaMessage(content="answer")
            raise RuntimeError("synthetic")

        async def generate_async():
            yield ChatCompletionDeltaMessage(content="answer")
            raise RuntimeError("synthetic")

        result = AsyncCompletionStream(generate_async(), evidence) if is_async else CompletionStream(generate(), evidence)
        if is_async:
            await anext(result)
            if end == "close":
                await result.aclose()
            else:
                with pytest.raises(RuntimeError, match="synthetic"):
                    await anext(result)
            with pytest.raises(StopAsyncIteration):
                await anext(result)
        else:
            next(result)
            if end == "close":
                result.close()
            else:
                with pytest.raises(RuntimeError, match="synthetic"):
                    next(result)
            with pytest.raises(StopIteration):
                next(result)
        assert result.metadata.stream_exhausted is False

    asyncio.run(run())
