from __future__ import annotations

import json
import logging

from vv_llm.chat_clients import ToolCallContentProcessor


def test_extract_tool_calls_and_non_tool_content() -> None:
    content = 'Need tools first.\n<|▶|>{"name": "get_weather", "arguments": {"city": "Paris"}}<|◀|>\n<|▶|>{"name": "get_time", "arguments": {"timezone": "UTC"}}<|◀|>'

    processor = ToolCallContentProcessor(content)
    parsed = processor.tool_calls

    assert processor.non_tool_content == "Need tools first."
    assert "tool_calls" in parsed
    assert len(parsed["tool_calls"]) == 2

    first_call = parsed["tool_calls"][0]
    assert first_call["id"].startswith("tool_call_")
    assert first_call["function"]["name"] == "get_weather"
    assert json.loads(first_call["function"]["arguments"]) == {"city": "Paris"}


def test_invalid_tool_call_payload_logs_warning(caplog) -> None:
    content = "<|▶|>{invalid json}<|◀|>"

    with caplog.at_level(logging.WARNING):
        parsed = ToolCallContentProcessor(content).tool_calls

    assert parsed == {}
    assert "Failed to parse tool call data" in caplog.text


def test_content_without_markers_returns_empty_tool_calls() -> None:
    processor = ToolCallContentProcessor("plain response")

    assert processor.tool_calls == {}
    assert processor.non_tool_content == "plain response"
