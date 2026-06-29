from typing import Final

from .base import ModelSettingDict


# 百度文心一言 ERNIE 模型
ERNIE_DEFAULT_MODEL: Final[str] = "ernie-4.5-8k-preview"
ERNIE_MODELS: Final[dict[str, ModelSettingDict]] = {
    "ernie-lite": {
        "id": "ernie-lite",
        "context_length": 6144,
        "max_output_tokens": 2048,
        "function_call_available": False,
        "response_format_available": False,
        "native_multimodal": False,
    },
    "ernie-speed": {
        "id": "ernie-speed",
        "context_length": 126976,
        "max_output_tokens": 4096,
        "function_call_available": False,
        "response_format_available": False,
        "native_multimodal": False,
    },
    "ernie-speed-pro-128k": {
        "id": "ernie-speed-pro-128k",
        "context_length": 126976,
        "max_output_tokens": 4096,
        "function_call_available": False,
        "response_format_available": False,
        "native_multimodal": False,
    },
    "ernie-3.5-8k": {
        "id": "ernie-3.5-8k",
        "context_length": 5120,
        "max_output_tokens": 2048,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "ernie-3.5-128k": {
        "id": "ernie-3.5-128k",
        "context_length": 126976,
        "max_output_tokens": 8192,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "ernie-4.0-8k-latest": {
        "id": "ernie-4.0-8k-latest",
        "context_length": 5120,
        "max_output_tokens": 2048,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "ernie-4.0-8k": {
        "id": "ernie-4.0-8k",
        "context_length": 5120,
        "max_output_tokens": 2048,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "ernie-4.0-turbo-8k": {
        "id": "ernie-4.0-turbo-8k",
        "context_length": 5120,
        "max_output_tokens": 2048,
        "function_call_available": False,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "ernie-4.5-8k-preview": {
        "id": "ernie-4.5-8k-preview",
        "context_length": 5120,
        "max_output_tokens": 2048,
        "function_call_available": False,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "ernie-4.5-turbo-128k": {
        "id": "ernie-4.5-turbo-128k",
        "context_length": 128000,
        "max_output_tokens": 65536,
        "function_call_available": False,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "ernie-5.0": {
        "id": "ernie-5.0",
        "context_length": 128000,
        "max_output_tokens": 65536,
        "function_call_available": False,
        "response_format_available": True,
        "native_multimodal": True,
    },
    "ernie-5.1": {
        "id": "ernie-5.1",
        "context_length": 128000,
        "max_output_tokens": 65536,
        "function_call_available": False,
        "response_format_available": True,
        "native_multimodal": False,
    },
}
