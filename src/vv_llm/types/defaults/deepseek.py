from typing import Final

from .base import ModelSettingDict


# Deepseek models
DEEPSEEK_DEFAULT_MODEL: Final[str] = "deepseek-v4-pro"
DEEPSEEK_MODELS: Final[dict[str, ModelSettingDict]] = {
    "deepseek-chat": {
        "id": "deepseek-chat",
        "context_length": 64000,
        "max_output_tokens": 8192,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "deepseek-reasoner": {
        "id": "deepseek-reasoner",
        "context_length": 64000,
        "max_output_tokens": 8192,
        "function_call_available": True,
        "response_format_available": False,
        "native_multimodal": False,
    },
    "deepseek-v4-flash": {
        "id": "deepseek-v4-flash",
        "context_length": 1000000,
        "max_output_tokens": 384000,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "deepseek-v4-pro": {
        "id": "deepseek-v4-pro",
        "context_length": 1000000,
        "max_output_tokens": 384000,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
}
