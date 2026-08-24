"""Groq defaults exposed from the versioned contract catalog."""

from typing import Final

from ._catalog import load_backend_defaults
from .base import ModelSettingDict

GROQ_DEFAULT_MODEL: Final[str]
GROQ_MODELS: Final[dict[str, ModelSettingDict]]
GROQ_DEFAULT_MODEL, GROQ_MODELS = load_backend_defaults("groq")
