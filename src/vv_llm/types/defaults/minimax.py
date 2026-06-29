from typing import Final

from .base import ModelSettingDict


# Minimax models
MINIMAX_DEFAULT_MODEL: Final[str] = "MiniMax-M3"
MINIMAX_MODELS: Final[dict[str, ModelSettingDict]] = {
    "MiniMax-M2.1": {
        "id": "MiniMax-M2.1",
        "context_length": 204800,
        "max_output_tokens": 10240,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "MiniMax-M2.1-lightning": {
        "id": "MiniMax-M2.1-lightning",
        "context_length": 204800,
        "max_output_tokens": 10240,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "MiniMax-M2.1-highspeed": {
        "id": "MiniMax-M2.1-highspeed",
        "context_length": 204800,
        "max_output_tokens": 10240,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "MiniMax-M2.5": {
        "id": "MiniMax-M2.5",
        "context_length": 204800,
        "max_output_tokens": 10240,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "MiniMax-M2.5-highspeed": {
        "id": "MiniMax-M2.5-highspeed",
        "context_length": 204800,
        "max_output_tokens": 10240,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "MiniMax-M2.7": {
        "id": "MiniMax-M2.7",
        "context_length": 204800,
        "max_output_tokens": 10240,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "MiniMax-M3": {
        "id": "MiniMax-M3",
        "context_length": 1_000_000,
        "max_output_tokens": 10240,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
    "MiniMax-M2.7-highspeed": {
        "id": "MiniMax-M2.7-highspeed",
        "context_length": 204800,
        "max_output_tokens": 10240,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
}
