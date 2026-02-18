from __future__ import annotations

import pytest

from vv_llm.chat_clients.message_normalizer import process_thinking_content


@pytest.mark.parametrize(
    ("raw", "in_reasoning", "start_tag", "end_tag", "expected_output", "expected_reasoning", "expected_state"),
    [
        ("<thought>thinking here</thought>actual content", False, "<thought>", "</thought>", "actual content", "thinking here", False),
        ("just regular content", False, "<thought>", "</thought>", "just regular content", "", False),
        ("</thought>", True, "<thought>", "</thought>", "", "", False),
        ("</thought>actual content", True, "<thought>", "</thought>", "actual content", "", False),
        ("<think>reason</think>answer", False, "<think>", "</think>", "answer", "reason", False),
    ],
)
def test_process_thinking_content_cases(
    raw: str,
    in_reasoning: bool,
    start_tag: str,
    end_tag: str,
    expected_output: str,
    expected_reasoning: str,
    expected_state: bool,
) -> None:
    buffer, output, reasoning, new_state = process_thinking_content(raw, in_reasoning, start_tag, end_tag)

    assert buffer == ""
    assert output == expected_output
    assert reasoning == expected_reasoning
    assert new_state is expected_state


def test_streaming_sequence_preserves_reasoning_state() -> None:
    state = False
    all_output = []
    all_reasoning = []

    for chunk in ["<thought>step1", " step2", "</thought>", "final answer"]:
        buffer, output, reasoning, state = process_thinking_content(chunk, state, "<thought>", "</thought>")
        assert buffer == ""
        all_output.append(output)
        all_reasoning.append(reasoning)

    assert "".join(all_reasoning) == "step1 step2"
    assert "".join(all_output) == "final answer"
    assert state is False
