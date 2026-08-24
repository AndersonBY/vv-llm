"""Gemini defaults exposed from the versioned contract catalog."""

from typing import Final

from ._catalog import load_backend_defaults
from .base import ModelSettingDict

GEMINI_DEFAULT_MODEL: Final[str]
GEMINI_MODELS: Final[dict[str, ModelSettingDict]]
GEMINI_DEFAULT_MODEL, GEMINI_MODELS = load_backend_defaults("gemini")
