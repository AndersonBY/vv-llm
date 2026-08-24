"""Mistral defaults exposed from the versioned contract catalog."""

from typing import Final

from ._catalog import load_backend_defaults
from .base import ModelSettingDict

MISTRAL_DEFAULT_MODEL: Final[str]
MISTRAL_MODELS: Final[dict[str, ModelSettingDict]]
MISTRAL_DEFAULT_MODEL, MISTRAL_MODELS = load_backend_defaults("mistral")
