from typing import Final

from .base import ModelSettingDict


# Gemini models
GEMINI_DEFAULT_MODEL: Final[str] = "gemini-3.5-flash"
GEMINI_MODELS: Final[dict[str, ModelSettingDict]] = {
    "gemini-2.5-pro": {
        "id": "gemini-2.5-pro",
        "context_length": 1048576,
        "max_output_tokens": 65536,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
    "gemini-2.5-flash": {
        "id": "gemini-2.5-flash",
        "context_length": 1048576,
        "max_output_tokens": 65536,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
    "gemini-2.5-flash-lite": {
        "id": "gemini-2.5-flash-lite",
        "context_length": 1048576,
        "max_output_tokens": 65536,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
    "gemini-3-pro-preview": {
        "id": "gemini-3-pro-preview",
        "context_length": 1048576,
        "max_output_tokens": 65536,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
    "gemini-3-pro": {
        "id": "gemini-3-pro",
        "context_length": 1048576,
        "max_output_tokens": 65536,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
    "gemini-3-pro-image-preview": {
        "id": "gemini-3-pro-image-preview",
        "context_length": 65536,
        "max_output_tokens": 32768,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
    "gemini-3-flash-preview": {
        "id": "gemini-3-flash-preview",
        "context_length": 1048576,
        "max_output_tokens": 65536,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
    "gemini-3-flash": {
        "id": "gemini-3-flash",
        "context_length": 1048576,
        "max_output_tokens": 65536,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
    "gemini-3.1-pro-preview": {
        "id": "gemini-3.1-pro-preview",
        "context_length": 1048576,
        "max_output_tokens": 65536,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
    "gemini-3.1-flash-lite-preview": {
        "id": "gemini-3.1-flash-lite-preview",
        "context_length": 1048576,
        "max_output_tokens": 65536,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
    "gemini-3.1-flash-lite": {
        "id": "gemini-3.1-flash-lite",
        "context_length": 1048576,
        "max_output_tokens": 65536,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
    "gemini-3.5-flash": {
        "id": "gemini-3.5-flash",
        "context_length": 1048576,
        "max_output_tokens": 65536,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": True,
    },
}
