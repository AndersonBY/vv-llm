from typing import Final

from .base import ModelSettingDict


# Moonshot models
MOONSHOT_DEFAULT_MODEL: Final[str] = "kimi-k2.6"
MOONSHOT_MODELS: Final[dict[str, ModelSettingDict]] = {
    "kimi-k2-0711-preview": {
        "id": "kimi-k2-0711-preview",
        "context_length": 131072,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "kimi-k2-0905-preview": {
        "id": "kimi-k2-0905-preview",
        "context_length": 256000,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "kimi-k2-turbo-preview": {
        "id": "kimi-k2-turbo-preview",
        "context_length": 131072,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "kimi-k2-thinking": {
        "id": "kimi-k2-thinking",
        "context_length": 256000,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "kimi-k2-thinking-turbo": {
        "id": "kimi-k2-thinking-turbo",
        "context_length": 256000,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "kimi-k2.5": {
        "id": "kimi-k2.5",
        "context_length": 256000,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
    "kimi-k2.6": {
        "id": "kimi-k2.6",
        "context_length": 256000,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
    "kimi-k2.7-code": {
        "id": "kimi-k2.7-code",
        "context_length": 256000,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
}
