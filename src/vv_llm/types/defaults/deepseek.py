"""DeepSeek defaults exposed from the versioned contract catalog."""

from typing import Final

from ._catalog import load_backend_defaults
from .base import ModelSettingDict

DEEPSEEK_DEFAULT_MODEL: Final[str]
DEEPSEEK_MODELS: Final[dict[str, ModelSettingDict]]
DEEPSEEK_DEFAULT_MODEL, DEEPSEEK_MODELS = load_backend_defaults("deepseek")
