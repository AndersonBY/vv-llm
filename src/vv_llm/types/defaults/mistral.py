from typing import Final

from .base import ModelSettingDict


# Mistral models
MISTRAL_DEFAULT_MODEL: Final[str] = "mistral-small"
MISTRAL_MODELS: Final[dict[str, ModelSettingDict]] = {
    "open-mistral-7b": {
        "id": "open-mistral-7b",
        "context_length": 32000,
        "function_call_available": False,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "open-mixtral-8x7b": {
        "id": "open-mixtral-8x7b",
        "context_length": 32000,
        "function_call_available": False,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "open-mixtral-8x22b": {
        "id": "open-mixtral-8x22b",
        "context_length": 64000,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "open-mistral-nemo": {
        "id": "open-mistral-nemo",
        "context_length": 128000,
        "function_call_available": False,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "codestral-latest": {
        "id": "codestral-latest",
        "context_length": 32000,
        "function_call_available": False,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "mistral-small-latest": {
        "id": "mistral-small-latest",
        "context_length": 30000,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "mistral-medium-latest": {
        "id": "mistral-medium-latest",
        "context_length": 30000,
        "function_call_available": False,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "mistral-large-latest": {
        "id": "mistral-large-latest",
        "context_length": 128000,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
}
