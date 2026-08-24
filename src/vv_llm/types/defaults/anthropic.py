"""Anthropic defaults exposed from the versioned contract catalog."""

from typing import Final

from ._catalog import load_backend_defaults
from .base import ModelSettingDict

ANTHROPIC_DEFAULT_MODEL: Final[str]
ANTHROPIC_MODELS: Final[dict[str, ModelSettingDict]]
ANTHROPIC_DEFAULT_MODEL, ANTHROPIC_MODELS = load_backend_defaults("anthropic")
