"""Reasoning controls shared by the existing provider adapters."""

import re
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
        if backend == "gemini" or uses_gemini_thinking_levels(model):
            google = result.get("google", {})
            nested = result.get("extra_body", {})
            configs = [google, nested.get("google", {})] if isinstance(nested, Mapping) else [google]
            for config in configs:
                thinking = config.get("thinking_config", {}) if isinstance(config, Mapping) else {}
                if isinstance(thinking, Mapping) and any(key in thinking for key in ("thinking_level", "thinking_budget", "thinkingLevel", "thinkingBudget")):
                    raise ValueError("reasoning_effort conflicts with Gemini thinking_level/thinking_budget")
        result = merge_reasoning_body(result, {container: {"effort": effort}} if container else {"reasoning_effort": effort})
    return effort, result


def uses_gemini_thinking_levels(model: str | None) -> bool:
    match = re.match(r"gemini-(\d+)(?:[.-]|$)", (model or "").rsplit("/", 1)[-1], re.IGNORECASE)
    return bool(match and int(match[1]) >= 3)


def normalize_gemini_body(body: Mapping[str, Any] | None) -> dict[str, Any]:
    """Use Gemini 3+ defaults instead of deprecated sampling and token budgets."""
    result = dict(body or {})
    for key in ("temperature", "top_p", "top_k", "topP", "topK", "thinking_budget", "thinkingBudget"):
        result.pop(key, None)
    if "thinkingLevel" in result:
        level = result.pop("thinkingLevel")
        if "thinking_level" in result and result["thinking_level"] != level:
            raise ValueError("Conflicting Gemini thinking_level values")
        result["thinking_level"] = level
    for key in ("extra_body", "google", "thinking_config"):
        if isinstance(result.get(key), Mapping):
            result[key] = normalize_gemini_body(result[key])
    return result
