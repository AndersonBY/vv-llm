"""Reasoning controls shared by the existing provider adapters."""

from collections.abc import Mapping
from typing import Any

from anthropic._types import NotGiven as AnthropicNotGiven
from openai._types import NotGiven as OpenAINotGiven


def merge_reasoning_body(body: Mapping[str, Any] | None, additions: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body or {})
    for key, value in additions.items():
        if key in result and result[key] != value:
            if isinstance(result[key], Mapping) and isinstance(value, Mapping):
                result[key] = merge_reasoning_body(result[key], value)
                continue
            raise ValueError(f"Conflicting reasoning control: {key}")
        result[key] = value
    return result


def resolve_reasoning_effort(effort: Any, body: Mapping[str, Any] | None, protocol: str, backend: str, model: str | None = None) -> tuple[str | None, dict[str, Any]]:
    result = dict(body or {})
    if model is not None and "model" in result and result["model"] != model:
        raise ValueError("Conflicting model values")
    if effort is not None and not isinstance(effort, (str, AnthropicNotGiven, OpenAINotGiven)):
        raise ValueError("reasoning_effort must be a non-empty string")
    effort = effort if isinstance(effort, str) else None
    controls = [effort] if effort is not None else []
    if "reasoning_effort" in result:
        controls.append(result.pop("reasoning_effort"))
    container = "reasoning" if protocol == "responses" else "output_config" if protocol == "anthropic" else None
    containers = ("reasoning", "output_config") if protocol == "validation" else (container,) if container else ()
    for field in containers:
        if isinstance(result.get(field), Mapping) and "effort" in result[field]:
            controls.append(result[field]["effort"])
    if any(not isinstance(value, str) or not value.strip() for value in controls):
        raise ValueError("reasoning_effort must be a non-empty string")
    if len(set(controls)) > 1:
        raise ValueError("Conflicting reasoning_effort values")
    effort = controls[0] if controls else None
    if effort is not None:
        if backend == "gemini":
            google = result.get("google", {})
            nested = result.get("extra_body", {})
            configs = [google, nested.get("google", {})] if isinstance(nested, Mapping) else [google]
            for config in configs:
                thinking = config.get("thinking_config", {}) if isinstance(config, Mapping) else {}
                if isinstance(thinking, Mapping) and ("thinking_level" in thinking or "thinking_budget" in thinking):
                    raise ValueError("reasoning_effort conflicts with Gemini thinking_level/thinking_budget")
        result = merge_reasoning_body(result, {container: {"effort": effort}} if container else {"reasoning_effort": effort})
    return effort, result
