from typing import Final

from .base import ModelSettingDict


# Groq models
GROQ_DEFAULT_MODEL: Final[str] = "llama3-70b-8192"
GROQ_MODELS: Final[dict[str, ModelSettingDict]] = {
    "mixtral-8x7b-32768": {
        "id": "mixtral-8x7b-32768",
        "context_length": 32768,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "llama3-70b-8192": {
        "id": "llama3-70b-8192",
        "context_length": 8192,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "llama3-8b-8192": {
        "id": "llama3-8b-8192",
        "context_length": 8192,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "gemma-7b-it": {
        "id": "gemma-7b-it",
        "context_length": 8192,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "gemma2-9b-it": {
        "id": "gemma2-9b-it",
        "context_length": 8192,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "llama3-groq-70b-8192-tool-use-preview": {
        "id": "llama3-groq-70b-8192-tool-use-preview",
        "context_length": 8192,
        "function_call_available": True,
        "response_format_available": True,
        "max_output_tokens": 8000,
        "native_multimodal": False,
    },
    "llama3-groq-8b-8192-tool-use-preview": {
        "id": "llama3-groq-8b-8192-tool-use-preview",
        "context_length": 8192,
        "function_call_available": True,
        "response_format_available": True,
        "max_output_tokens": 8000,
        "native_multimodal": False,
    },
    "llama-3.1-70b-versatile": {
        "id": "llama-3.1-70b-versatile",
        "context_length": 131072,
        "function_call_available": True,
        "response_format_available": True,
        "max_output_tokens": 8000,
        "native_multimodal": False,
    },
    "llama-3.1-8b-instant": {
        "id": "llama-3.1-8b-instant",
        "context_length": 131072,
        "function_call_available": True,
        "response_format_available": True,
        "max_output_tokens": 8000,
        "native_multimodal": False,
    },
}
