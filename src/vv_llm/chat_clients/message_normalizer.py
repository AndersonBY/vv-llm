import json
import re
from collections.abc import Iterable
from typing import Any

from anthropic.types import (
    MessageParam,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)

from ..types.enums import BackendType


def is_gemini_3_model(model_id: str | None, backend: BackendType) -> bool:
    """Check if the current model is a Gemini 3 series model."""
    return backend == BackendType.Gemini and bool(model_id and str(model_id).startswith("gemini-3"))


def get_thinking_tags(backend: BackendType) -> tuple[str, str]:
    """Get the thinking tags used by the backend."""
    if backend == BackendType.Gemini:
        return ("<thought>", "</thought>")
    return ("<think>", "</think>")


def process_thinking_content(buffer: str, in_reasoning: bool, start_tag: str, end_tag: str) -> tuple[str, str, str, bool]:
    """Split streamed content into normal output and reasoning output."""
    current_output_content = ""
    current_reasoning_content = ""

    while buffer:
        if not in_reasoning:
            start_pos = buffer.find(start_tag)
            if start_pos != -1:
                if start_pos > 0:
                    current_output_content += buffer[:start_pos]
                buffer = buffer[start_pos + len(start_tag) :]
                in_reasoning = True
            else:
                current_output_content += buffer
                buffer = ""
        else:
            end_pos = buffer.find(end_tag)
            if end_pos != -1:
                current_reasoning_content += buffer[:end_pos]
                buffer = buffer[end_pos + len(end_tag) :]
                in_reasoning = False
            else:
                current_reasoning_content += buffer
                buffer = ""

    return buffer, current_output_content, current_reasoning_content, in_reasoning


def process_reasoning_content(buffer: str, in_reasoning: bool, start_tag: str, end_tag: str) -> tuple[str, str, bool]:
    """Normalize streamed reasoning deltas and keep thought-tag state in sync."""
    current_reasoning_content = ""

    while buffer:
        if not in_reasoning:
            start_pos = buffer.find(start_tag)
            end_pos = buffer.find(end_tag)

            if start_pos != -1 and (end_pos == -1 or start_pos < end_pos):
                if start_pos > 0:
                    current_reasoning_content += buffer[:start_pos]
                buffer = buffer[start_pos + len(start_tag) :]
                in_reasoning = True
            elif end_pos != -1:
                if end_pos > 0:
                    current_reasoning_content += buffer[:end_pos]
                buffer = buffer[end_pos + len(end_tag) :]
            else:
                current_reasoning_content += buffer
                buffer = ""
        else:
            end_pos = buffer.find(end_tag)
            if end_pos != -1:
                current_reasoning_content += buffer[:end_pos]
                buffer = buffer[end_pos + len(end_tag) :]
                in_reasoning = False
            else:
                current_reasoning_content += buffer
                buffer = ""

    return buffer, current_reasoning_content, in_reasoning


def extract_reasoning_tagged_content(content: str) -> tuple[str, str | None]:
    """Extract reasoning content wrapped by think tags from a full response."""
    for start_tag, end_tag in (("<think>", "</think>"), ("<thought>", "</thought>")):
        think_pattern = f"{re.escape(start_tag)}(.*?){re.escape(end_tag)}"
        think_matches = re.findall(think_pattern, content, re.DOTALL)
        if think_matches:
            reasoning = "".join(think_matches)
            cleaned_content = re.sub(think_pattern, "", content, flags=re.DOTALL)
            return cleaned_content, reasoning
    return content, None


def strip_reasoning_tags(content: str | None, start_tag: str, end_tag: str) -> str | None:
    """Remove backend-specific reasoning tags from reasoning text blocks."""
    if content is None:
        return None
    return content.replace(start_tag, "").replace(end_tag, "")


def format_messages_alternate(messages: list) -> list:
    """Merge adjacent same-role messages into Anthropic-compatible alternation."""
    formatted_messages = []
    current_role = None
    current_content = []

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role != current_role:
            if current_content:
                formatted_messages.append({"role": current_role, "content": current_content})
                current_content = []
            current_role = role

        if isinstance(content, str):
            text_block = {"type": "text", "text": content}
            if "cache_control" in message:
                text_block["cache_control"] = message["cache_control"]
            current_content.append(text_block)
        elif isinstance(content, list):
            current_content.extend(content)
        else:
            current_content.append(content)

    if current_content:
        formatted_messages.append({"role": current_role, "content": current_content})

    return formatted_messages


def refactor_into_openai_messages(messages: Iterable[MessageParam]):
    """Convert Anthropic blocks into OpenAI-like message format."""
    formatted_messages = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            formatted_messages.append(message)
        elif isinstance(content, list):
            _content = []
            for item in content:
                if isinstance(item, TextBlock | ToolUseBlock):
                    _content.append(item.model_dump())
                elif isinstance(item, ThinkingBlock | RedactedThinkingBlock):
                    continue
                elif isinstance(item, dict) and item.get("type") == "image":
                    source = item.get("source", {})
                    image_data = source.get("data", "") if isinstance(source, dict) else ""
                    media_type = source.get("media_type", "") if isinstance(source, dict) else ""
                    data_url = f"data:{media_type};base64,{image_data}"
                    _content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                                "detail_type": "auto",
                            },
                        }
                    )
                else:
                    _content.append(item)
            formatted_messages.append({"role": message["role"], "content": _content})
        else:
            formatted_messages.append(message)
    return formatted_messages


def _response_api_content(content: Any) -> Any:
    """Convert Chat Completions content parts into Responses API input parts."""

    if not isinstance(content, list):
        return content
    converted: list[Any] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            converted.append({"type": "input_text", "text": part.get("text", "")})
        elif isinstance(part, dict) and part.get("type") == "image_url":
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            converted.append({"type": "input_image", "image_url": url})
        else:
            converted.append(part)
    return converted


def messages_for_response_api(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Chat Completions messages into Responses API input items.

    Assistant ``tool_calls`` become ``function_call`` items and ``tool`` results
    become ``function_call_output`` items, both correlated by ``call_id``.
    """

    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "tool":
            output = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id"),
                    "output": output,
                }
            )
            continue
        if role == "assistant" and message.get("tool_calls"):
            if content:
                items.append({"role": "assistant", "content": _response_api_content(content)})
            for tool_call in message["tool_calls"]:
                function = tool_call.get("function") or {}
                arguments = function.get("arguments")
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.get("id"),
                        "name": function.get("name"),
                        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {}, ensure_ascii=False),
                    }
                )
            continue
        items.append({"role": role, "content": _response_api_content("" if content is None else content)})
    return items
