import json
from collections.abc import Iterable
from typing import Any, cast

from openai.types.responses.function_tool_param import FunctionToolParam as ResponsesFunctionToolParam

from ..types.llm_parameters import AnthropicToolChoice, AnthropicToolParam, ToolChoice, ToolParam


def tool_schema_for_response_api(tool: ToolParam) -> ResponsesFunctionToolParam:
    tool_data = cast(dict[str, Any], tool)
    function_data = cast(dict[str, Any], tool_data["function"])
    return {
        "name": function_data["name"],
        "description": function_data.get("description", ""),
        "parameters": function_data.get("parameters", {}),
        "strict": function_data.get("strict", False),
        "type": "function",
    }


def refactor_tool_use_params(tools: Iterable[ToolParam]) -> list[AnthropicToolParam]:
    result = []
    for tool in tools:
        tool_data = cast(dict[str, Any], tool)
        function_data = cast(dict[str, Any], tool_data["function"])
        tool_param: AnthropicToolParam = {
            "name": function_data["name"],
            "description": function_data.get("description", ""),
            "input_schema": function_data.get("parameters", {}),
        }
        if "cache_control" in tool_data:
            cast(dict[str, Any], tool_param)["cache_control"] = tool_data["cache_control"]
        elif "cache_control" in function_data:
            cast(dict[str, Any], tool_param)["cache_control"] = function_data["cache_control"]
        result.append(tool_param)
    return result


def refactor_tool_calls(tool_calls: list):
    return [
        {
            "index": index,
            "id": tool["id"],
            "type": "function",
            "function": {
                "name": tool["name"],
                "arguments": json.dumps(tool["input"], ensure_ascii=False),
            },
        }
        for index, tool in enumerate(tool_calls)
    ]


def refactor_tool_choice(tool_choice: ToolChoice) -> AnthropicToolChoice:
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "required":
            return {"type": "any"}
    elif isinstance(tool_choice, dict) and "function" in tool_choice:
        tool_choice_data = cast(dict[str, Any], tool_choice)
        function_data = cast(dict[str, Any], tool_choice_data["function"])
        return {"type": "tool", "name": function_data["name"]}
    return {"type": "auto"}
