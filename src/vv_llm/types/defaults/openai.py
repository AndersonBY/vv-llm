"""OpenAI defaults exposed from the versioned contract catalog."""

from typing import Final

from ._catalog import load_backend_defaults
from .base import ModelSettingDict

OPENAI_DEFAULT_MODEL: Final[str]
OPENAI_MODELS: Final[dict[str, ModelSettingDict]]
OPENAI_DEFAULT_MODEL, OPENAI_MODELS = load_backend_defaults("openai")
