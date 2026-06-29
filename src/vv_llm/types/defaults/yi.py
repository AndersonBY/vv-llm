from typing import Final

from .base import ModelSettingDict


# Yi models
YI_DEFAULT_MODEL: Final[str] = "yi-lightning"
YI_MODELS: Final[dict[str, ModelSettingDict]] = {
    "yi-lightning": {
        "id": "yi-lightning",
        "context_length": 16000,
        "max_output_tokens": 4096,
        "function_call_available": False,
        "response_format_available": False,
        "native_multimodal": False,
    },
    "yi-vision-v2": {
        "id": "yi-vision-v2",
        "context_length": 16000,
        "max_output_tokens": 2000,
        "function_call_available": False,
        "response_format_available": False,
        "native_multimodal": True,
    },
}
