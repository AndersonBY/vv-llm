from typing import Final

from .base import ModelSettingDict


# Baichuan models
BAICHUAN_DEFAULT_MODEL: Final[str] = "Baichuan4"
BAICHUAN_MODELS: Final[dict[str, ModelSettingDict]] = {
    "Baichuan4": {
        "id": "Baichuan4",
        "context_length": 32768,
        "max_output_tokens": 2048,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
}
