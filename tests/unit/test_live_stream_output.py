from __future__ import annotations

from types import SimpleNamespace

from tests.live.live_common import print_stream_chunk


def _capture_printer():
    chunks: list[str] = []

    def printer(*args, **kwargs):
        text = " ".join(str(arg) for arg in args)
        end = kwargs.get("end", "\n")
        chunks.append(text + end)

    return chunks, printer


def test_print_stream_chunk_omits_none_values_from_reasoning_chunk() -> None:
    outputs, printer = _capture_printer()
    chunk = SimpleNamespace(
        reasoning_content="reasoning text",
        content=None,
        tool_calls=None,
        usage=None,
    )

    start_content = print_stream_chunk(chunk, use_tool=True, start_content=False, print_fn=printer)

    assert start_content is False
    assert "".join(outputs) == "reasoning text"


def test_print_stream_chunk_prints_content_tool_calls_and_usage_without_none_noise() -> None:
    outputs, printer = _capture_printer()
    chunk = SimpleNamespace(
        reasoning_content=None,
        content="final answer",
        tool_calls=["tool-call"],
        usage="usage-data",
    )

    start_content = print_stream_chunk(chunk, use_tool=True, start_content=False, print_fn=printer)

    assert start_content is True
    assert "".join(outputs) == "\n=== Content Start ===\n\nfinal answer['tool-call']\nUsage: usage-data\n====================\n"
