"""Qwen defaults exposed from the versioned contract catalog."""

from typing import Final

from ._catalog import load_backend_defaults
from .base import ModelSettingDict

QWEN_DEFAULT_MODEL: Final[str]
QWEN_MODELS: Final[dict[str, ModelSettingDict]]
QWEN_DEFAULT_MODEL, QWEN_MODELS = load_backend_defaults("qwen")
