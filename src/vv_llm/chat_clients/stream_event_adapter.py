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


def _response_tool_call_id(item: Any) -> str | None:
    """Prefer the provider ``call_id`` so tool results can be correlated."""

    return getattr(item, "call_id", None) or getattr(item, "id", None)


def _response_tool_call_delta(
    output_index: int,
    tool_call: dict[str, Any],
    arguments: str,
    *,
    include_header: bool,
    raw_content: dict[str, Any] | None = None,
) -> ChatCompletionDeltaMessage:
    """Emit one incremental tool-call delta; only the first delta carries id/name."""

    function: dict[str, Any] = {"arguments": arguments}
    call: dict[str, Any] = {"index": output_index, "type": "function", "function": function}
    if include_header:
        function["name"] = tool_call.get("name")
        call["id"] = tool_call.get("id")
    tool_call["started"] = True
    return ChatCompletionDeltaMessage(tool_calls=cast(Any, [call]), raw_content=raw_content)


def _response_tool_call_remainder(tool_call: dict[str, Any], full_arguments: str | None) -> str:
    """Return the arguments that were not streamed incrementally."""

    streamed = tool_call.get("arguments") or ""
    full = full_arguments or ""
    if not full or full == streamed:
        return ""
    remainder = full[len(streamed) :] if streamed and full.startswith(streamed) else full
    tool_call["arguments"] = streamed + remainder
    return remainder


def _gemini3_tool_call_raw_content(tool_call: dict[str, Any]) -> dict[str, Any] | None:
    extra_content = tool_call.get("extra_content")
    if isinstance(extra_content, dict) and extra_content.get("google"):
        return {"google": extra_content["google"]}
    return None


def adapt_response_api_stream_event(event: Any, final_tool_calls: dict[int, dict[str, Any]], is_gemini3: bool) -> tuple[list[ChatCompletionDeltaMessage], Usage | None]:
    """Convert a Responses API stream event into ChatCompletionDeltaMessage objects.

    Tool-call arguments are emitted exactly once: the first delta carries the
    call header, argument deltas carry their own fragment, and the terminal
    ``*.done`` events only append a fragment that was never streamed.
    """

    event_type = event.type

    if event_type == "response.output_text.delta":
        if event.delta:
            return [ChatCompletionDeltaMessage(content=event.delta)], None
        return [], None

    if event_type == "response.output_item.added":
        item = event.item
        output_index = event.output_index
        if item and item.type == "function_call" and output_index is not None:
            tool_call = final_tool_calls[output_index] = {
                "id": _response_tool_call_id(item),
                "name": item.name,
                "arguments": getattr(item, "arguments", None) or "",
                "extra_content": getattr(item, "extra_content", None),
                "started": False,
            }
            return [
                _response_tool_call_delta(
                    output_index,
                    tool_call,
                    tool_call["arguments"],
                    include_header=True,
                )
            ], None
        return [], None

    if event_type == "response.function_call_arguments.delta":
        output_index = event.output_index
        delta = event.delta
        if output_index is not None and delta is not None:
            tool_call = final_tool_calls.setdefault(output_index, {"id": None, "name": None, "arguments": "", "extra_content": None, "started": False})
            tool_call["arguments"] += delta
            return [
                _response_tool_call_delta(
                    output_index,
                    tool_call,
                    delta,
                    include_header=not tool_call["started"],
                )
            ], None
        return [], None

    if event_type == "response.function_call_arguments.done":
        output_index = event.output_index
        if output_index is not None:
            tool_call = final_tool_calls.setdefault(output_index, {"id": None, "name": None, "arguments": "", "extra_content": None, "started": False})
            raw_content = _gemini3_tool_call_raw_content(tool_call) if is_gemini3 else None
            remainder = _response_tool_call_remainder(tool_call, getattr(event, "arguments", None))
            if remainder:
                return [
                    _response_tool_call_delta(
                        output_index,
                        tool_call,
                        remainder,
                        include_header=not tool_call["started"],
                        raw_content=raw_content,
                    )
                ], None
            if raw_content is not None:
                return [ChatCompletionDeltaMessage(raw_content=raw_content)], None
        return [], None

    if event_type == "response.output_item.done":
        item = event.item
        output_index = event.output_index
        if item and item.type == "function_call" and output_index is not None:
            tool_call = final_tool_calls.setdefault(
                output_index,
                {
                    "id": _response_tool_call_id(item),
                    "name": item.name,
                    "arguments": "",
                    "extra_content": getattr(item, "extra_content", None),
                    "started": False,
                },
            )
            remainder = _response_tool_call_remainder(tool_call, getattr(item, "arguments", None))
            if remainder:
                return [
                    _response_tool_call_delta(
                        output_index,
                        tool_call,
                        remainder,
                        include_header=not tool_call["started"],
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
        cache_creation_tokens = getattr(chunk.message.usage, "cache_creation_input_tokens", 0) or 0
        prompt_tokens = chunk.message.usage.input_tokens + cache_read_tokens + cache_creation_tokens
        usage_data: dict[str, Any] = {"prompt_tokens": prompt_tokens}
        if cache_read_tokens:
            usage_data["prompt_tokens_details"] = {"cached_tokens": cache_read_tokens}
        if cache_creation_tokens:
            usage_data["cache_creation_tokens"] = cache_creation_tokens
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
        if "cache_creation_tokens" in stream_state["usage"]:
            usage_kwargs["cache_creation_tokens"] = stream_state["usage"]["cache_creation_tokens"]
        return ChatCompletionDeltaMessage(usage=Usage(**usage_kwargs))

    if isinstance(chunk, RawMessageStopEvent | RawContentBlockStopEvent):
        return None

    return None


def build_anthropic_completion_message(content_blocks: list[Any], usage: Any) -> ChatCompletionMessage:
    cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
    prompt_tokens = usage.input_tokens + cache_read_tokens + cache_creation_tokens
    result: dict[str, Any] = {
        "content": "",
        "reasoning_content": "",
        "raw_content": [content_block.model_dump() for content_block in content_blocks],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": prompt_tokens + usage.output_tokens,
            "prompt_tokens_details": {
                "cached_tokens": cache_read_tokens,
            },
            "cache_creation_tokens": cache_creation_tokens,
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
