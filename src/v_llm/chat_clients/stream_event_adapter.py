import json
from typing import Any, cast

from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from openai.types.completion_usage import PromptTokensDetails

from ..types.llm_parameters import ChatCompletionDeltaMessage, ChatCompletionMessage, Usage
from .tool_call_parser import refactor_tool_calls


def adapt_response_api_stream_event(event: Any, final_tool_calls: dict[int, dict[str, Any]], is_gemini3: bool) -> tuple[list[ChatCompletionDeltaMessage], Usage | None]:
    """Convert a Responses API stream event into ChatCompletionDeltaMessage objects."""
    event_type = event.type

    if event_type == "response.output_text.delta":
        if event.delta:
            return [ChatCompletionDeltaMessage(content=event.delta)], None
        return [], None

    if event_type == "response.output_item.added":
        item = event.item
        output_index = event.output_index
        if item and item.type == "function_call" and output_index is not None:
            extra_content = getattr(item, "extra_content", None)
            final_tool_calls[output_index] = {
                "id": item.id,
                "call_id": item.call_id,
                "name": item.name,
                "arguments": item.arguments or "",
                "extra_content": extra_content,
            }
            return [
                ChatCompletionDeltaMessage(
                    tool_calls=cast(
                        Any,
                        [
                            {
                                "index": output_index,
                                "id": item.id,
                                "type": "function",
                                "function": {
                                    "name": item.name,
                                    "arguments": "",
                                },
                            }
                        ],
                    ),
                )
            ], None
        return [], None

    if event_type == "response.function_call_arguments.delta":
        output_index = event.output_index
        delta = event.delta
        if output_index is not None and delta is not None:
            if output_index in final_tool_calls:
                final_tool_calls[output_index]["arguments"] += delta
                call_id = final_tool_calls[output_index].get("id")
            else:
                call_id = None
            return [
                ChatCompletionDeltaMessage(
                    tool_calls=cast(
                        Any,
                        [
                            {
                                "index": output_index,
                                "id": call_id,
                                "type": "function",
                                "function": {"arguments": delta},
                            }
                        ],
                    ),
                )
            ], None
        return [], None

    if event_type == "response.function_call_arguments.done":
        output_index = event.output_index
        if output_index is not None and output_index in final_tool_calls:
            tool_call = final_tool_calls[output_index]
            raw_content = None
            if is_gemini3:
                extra_content = tool_call.get("extra_content")
                if isinstance(extra_content, dict) and extra_content.get("google"):
                    raw_content = {"google": extra_content["google"]}
            return [
                ChatCompletionDeltaMessage(
                    tool_calls=cast(
                        Any,
                        [
                            {
                                "index": output_index,
                                "id": tool_call.get("id"),
                                "type": "function",
                                "function": {
                                    "name": tool_call.get("name"),
                                    "arguments": tool_call.get("arguments", ""),
                                },
                            }
                        ],
                    ),
                    raw_content=raw_content,
                )
            ], None
        return [], None

    if event_type == "response.output_item.done":
        item = event.item
        output_index = event.output_index
        if item and item.type == "function_call" and output_index is not None:
            return [
                ChatCompletionDeltaMessage(
                    tool_calls=cast(
                        Any,
                        [
                            {
                                "index": output_index,
                                "id": item.id,
                                "type": "function",
                                "function": {
                                    "name": item.name,
                                    "arguments": item.arguments,
                                },
                            }
                        ],
                    ),
                )
            ], None
        return [], None

    if event_type == "response.completed":
        final_resp = event.response
        if final_resp and final_resp.usage:
            usage = final_resp.usage
            return [], Usage(
                completion_tokens=usage.output_tokens or 0,
                prompt_tokens=usage.input_tokens or 0,
                total_tokens=(usage.input_tokens or 0) + (usage.output_tokens or 0),
            )
        return [], None

    if event_type in ("response.error", "error"):
        raise RuntimeError(f"Responses stream error: {event.error}")

    return [], None


def init_anthropic_stream_state() -> dict[str, Any]:
    return {
        "content": "",
        "reasoning_content": "",
        "usage": {},
        "tool_calls": [],
        "raw_content": [],
    }


def _last_raw_content(stream_state: dict[str, Any], block_type: str) -> dict[str, Any] | None:
    for i in range(len(stream_state["raw_content"]) - 1, -1, -1):
        item = stream_state["raw_content"][i]
        if item["type"] == block_type:
            return cast(dict[str, Any], item)
    return None


def adapt_anthropic_stream_event(chunk: Any, stream_state: dict[str, Any]) -> ChatCompletionDeltaMessage | None:
    message: dict[str, Any] = {"content": "", "tool_calls": []}

    if isinstance(chunk, RawMessageStartEvent):
        cache_read_tokens = getattr(chunk.message.usage, "cache_read_input_tokens", 0) or 0
        prompt_tokens = chunk.message.usage.input_tokens + cache_read_tokens
        usage_data: dict[str, Any] = {"prompt_tokens": prompt_tokens}
        if cache_read_tokens:
            usage_data["prompt_tokens_details"] = {"cached_tokens": cache_read_tokens}
        stream_state["usage"] = usage_data
        return None

    if isinstance(chunk, RawContentBlockStartEvent):
        content_block = chunk.content_block.model_dump()
        stream_state["raw_content"].append(content_block)
        if chunk.content_block.type == "tool_use":
            stream_state["tool_calls"] = message["tool_calls"] = [
                {
                    "index": 0,
                    "id": chunk.content_block.id,
                    "function": {
                        "arguments": "",
                        "name": chunk.content_block.name,
                    },
                    "type": "function",
                }
            ]
        elif chunk.content_block.type == "text":
            message["content"] = chunk.content_block.text
        elif chunk.content_block.type == "thinking":
            message["reasoning_content"] = chunk.content_block.thinking
        message["raw_content"] = content_block
        return ChatCompletionDeltaMessage(**message)

    if isinstance(chunk, RawContentBlockDeltaEvent):
        if chunk.delta.type == "text_delta":
            message["content"] = chunk.delta.text
            stream_state["content"] += chunk.delta.text
            text_block = _last_raw_content(stream_state, "text")
            if text_block is not None:
                text_block["text"] += chunk.delta.text
        elif chunk.delta.type == "thinking_delta":
            message["reasoning_content"] = chunk.delta.thinking
            stream_state["reasoning_content"] += chunk.delta.thinking
            thinking_block = _last_raw_content(stream_state, "thinking")
            if thinking_block is not None:
                thinking_block["thinking"] += chunk.delta.thinking
        elif chunk.delta.type == "signature_delta":
            thinking_block = _last_raw_content(stream_state, "thinking")
            if thinking_block is not None:
                if "signature" not in thinking_block:
                    thinking_block["signature"] = ""
                thinking_block["signature"] += chunk.delta.signature
        elif chunk.delta.type == "citations_delta":
            citation_data = chunk.delta.citation.model_dump()
            raw_content = _last_raw_content(stream_state, stream_state["raw_content"][-1]["type"]) if stream_state["raw_content"] else None
            if raw_content is not None:
                if "citations" not in raw_content:
                    raw_content["citations"] = []
                raw_content["citations"].append(citation_data)
        elif chunk.delta.type == "input_json_delta" and stream_state["tool_calls"]:
            stream_state["tool_calls"][0]["function"]["arguments"] += chunk.delta.partial_json
            message["tool_calls"] = [
                {
                    "index": 0,
                    "id": stream_state["tool_calls"][0]["id"],
                    "function": {
                        "arguments": chunk.delta.partial_json,
                        "name": stream_state["tool_calls"][0]["function"]["name"],
                    },
                    "type": "function",
                }
            ]
            tool_use_block = _last_raw_content(stream_state, "tool_use")
            if tool_use_block is not None:
                if "input" not in tool_use_block:
                    tool_use_block["input"] = {}
                try:
                    if stream_state["tool_calls"][0]["function"]["arguments"]:
                        tool_use_block["input"] = json.loads(stream_state["tool_calls"][0]["function"]["arguments"])
                    else:
                        tool_use_block["input"] = {}
                except json.JSONDecodeError:
                    pass
        elif chunk.delta.type == "redacted_thinking_delta":
            redacted_block = _last_raw_content(stream_state, "redacted_thinking")
            if redacted_block is not None:
                if "data" not in redacted_block:
                    redacted_block["data"] = ""
                redacted_block["data"] += chunk.delta.data

        message["raw_content"] = chunk.delta.model_dump()
        return ChatCompletionDeltaMessage(**message)

    if isinstance(chunk, RawMessageDeltaEvent):
        stream_state["usage"]["completion_tokens"] = chunk.usage.output_tokens
        stream_state["usage"]["total_tokens"] = stream_state["usage"]["prompt_tokens"] + stream_state["usage"]["completion_tokens"]
        usage_kwargs: dict[str, Any] = {
            "prompt_tokens": stream_state["usage"]["prompt_tokens"],
            "completion_tokens": stream_state["usage"]["completion_tokens"],
            "total_tokens": stream_state["usage"]["total_tokens"],
        }
        if "prompt_tokens_details" in stream_state["usage"]:
            usage_kwargs["prompt_tokens_details"] = PromptTokensDetails(cached_tokens=stream_state["usage"]["prompt_tokens_details"].get("cached_tokens", 0))
        return ChatCompletionDeltaMessage(usage=Usage(**usage_kwargs))

    if isinstance(chunk, RawMessageStopEvent | RawContentBlockStopEvent):
        return None

    return None


def build_anthropic_completion_message(content_blocks: list[Any], usage: Any) -> ChatCompletionMessage:
    result: dict[str, Any] = {
        "content": "",
        "reasoning_content": "",
        "raw_content": [content_block.model_dump() for content_block in content_blocks],
        "usage": {
            "prompt_tokens": usage.input_tokens + usage.cache_read_input_tokens if usage.cache_read_input_tokens else usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
            "prompt_tokens_details": {
                "cached_tokens": usage.cache_read_input_tokens,
            },
        },
    }
    tool_calls = []
    for content_block in content_blocks:
        if isinstance(content_block, TextBlock):
            result["content"] += content_block.text
        elif isinstance(content_block, ThinkingBlock):
            result["reasoning_content"] = content_block.thinking
        elif isinstance(content_block, ToolUseBlock):
            tool_calls.append(content_block.model_dump())

    if tool_calls:
        result["tool_calls"] = refactor_tool_calls(tool_calls)

    return ChatCompletionMessage(**result)
