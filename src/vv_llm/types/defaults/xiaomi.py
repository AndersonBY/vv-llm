from typing import Final

from .base import ModelSettingDict


XIAOMI_DEFAULT_MODEL: Final[str] = "mimo-v2-pro"
XIAOMI_MODELS: Final[dict[str, ModelSettingDict]] = {
    "mimo-v2-pro": {
        "id": "mimo-v2-pro",
        "context_length": 1000000,
        "max_output_tokens": 128000,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "mimo-v2-omni": {
        "id": "mimo-v2-omni",
        "context_length": 256000,
        "max_output_tokens": 128000,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
    "mimo-v2-tts": {
        "id": "mimo-v2-tts",
        "context_length": 8000,
        "max_output_tokens": 8000,
        "function_call_available": False,
        "response_format_available": False,
        "native_multimodal": False,
    },
    "mimo-v2-flash": {
        "id": "mimo-v2-flash",
        "context_length": 256000,
        "max_output_tokens": 64000,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
}
