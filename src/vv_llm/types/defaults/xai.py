from typing import Final

from .base import ModelSettingDict


XAI_DEFAULT_MODEL: Final[str] = "grok-4.20-0309-reasoning"
XAI_MODELS: Final[dict[str, ModelSettingDict]] = {
    "grok-4.20-0309-reasoning": {
        "id": "grok-4.20-0309-reasoning",
        "context_length": 2000000,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
    "grok-4.20-0309-non-reasoning": {
        "id": "grok-4.20-0309-non-reasoning",
        "context_length": 2000000,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
    "grok-4.20-multi-agent-0309": {
        "id": "grok-4.20-multi-agent-0309",
        "context_length": 2000000,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
    "grok-4-1-fast-reasoning": {
        "id": "grok-4-1-fast-reasoning",
        "context_length": 2000000,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
    "grok-4-1-fast-non-reasoning": {
        "id": "grok-4-1-fast-non-reasoning",
        "context_length": 2000000,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
}
