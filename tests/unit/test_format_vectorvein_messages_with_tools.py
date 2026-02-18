from __future__ import annotations

import json

import pytest

from vv_llm.chat_clients.utils import format_messages
from vv_llm.types.enums import BackendType


@pytest.fixture
def workflow_message() -> dict:
    return {
        "author_type": "A",
        "content_type": "WKF",
        "status": "S",
        "metadata": {
            "record_id": "record-1",
            "workflow_result": "ok",
            "selected_workflow": {
                "tool_call_id": "tool-from-selected",
                "function_name": "search_inventory",
                "params": {"company_id": "COM001"},
            },
            "tool_calls": [
                {
                    "id": "tool-from-preserved",
                    "type": "function",
                    "function": {
                        "name": "search_inventory",
                        "arguments": json.dumps({"company_id": "COM001"}),
                    },
                    "extra_content": {"google": {"thought_signature": "abc"}},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        "content": {"text": "Checking inventory now."},
        "attachments": [],
    }


def test_format_vectorvein_workflow_for_openai_style_backends(workflow_message: dict) -> None:
    messages = [
        {
            "author_type": "U",
            "content_type": "TXT",
            "content": {"text": "Find inventory"},
            "attachments": [],
        },
        workflow_message,
    ]

    formatted = format_messages(messages, backend=BackendType.OpenAI, native_multimodal=True, function_call_available=True, process_image=False)

    assert len(formatted) == 3
    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"] == "Find inventory"

    assistant_tool_call = formatted[1]["tool_calls"][0]
    assert formatted[1]["role"] == "assistant"
    assert formatted[1]["content"] == "Checking inventory now."
    assert assistant_tool_call["id"] == "tool-from-preserved"
    assert assistant_tool_call["function"]["name"] == "search_inventory"
    assert assistant_tool_call["extra_content"]["google"]["thought_signature"] == "abc"

    assert formatted[2]["role"] == "tool"
    assert formatted[2]["tool_call_id"] == "tool-from-preserved"
    assert formatted[2]["content"] == "ok"


def test_format_vectorvein_workflow_for_anthropic_backend(workflow_message: dict) -> None:
    workflow_message["metadata"].pop("tool_calls")

    formatted = format_messages([workflow_message], backend=BackendType.Anthropic, process_image=False)

    assert len(formatted) == 2
    assert formatted[0]["role"] == "assistant"
    assert formatted[0]["content"][0]["type"] == "text"
    assert formatted[0]["content"][1]["type"] == "tool_use"
    assert formatted[0]["content"][1]["id"] == "tool-from-selected"

    assert formatted[1]["role"] == "user"
    assert formatted[1]["content"][0]["type"] == "tool_result"
    assert formatted[1]["content"][0]["tool_use_id"] == "tool-from-selected"


def test_format_messages_rejects_unknown_vectorvein_message_type() -> None:
    with pytest.raises(ValueError, match="Unsupported message type"):
        format_messages(
            [
                {
                    "author_type": "U",
                    "content_type": "UNKNOWN",
                    "content": {"text": "bad"},
                    "attachments": [],
                }
            ],
            backend=BackendType.OpenAI,
        )
