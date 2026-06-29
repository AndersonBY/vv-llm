from typing import Final

from .base import ModelSettingDict


STEPFUN_DEFAULT_MODEL: Final[str] = "step-3.5-flash"
STEPFUN_MODELS: Final[dict[str, ModelSettingDict]] = {
    "step-2-mini": {
        "id": "step-2-mini",
        "context_length": 32768,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
    "step-3": {
        "id": "step-3",
        "context_length": 65536,
        "function_call_available": True,
        "response_format_available": False,
        "native_multimodal": True,
    },
    "step-3.5-flash": {
        "id": "step-3.5-flash",
        "context_length": 256000,
        "function_call_available": True,
        "response_format_available": True,
        "native_multimodal": False,
    },
}
