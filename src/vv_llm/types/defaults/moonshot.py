"""Moonshot defaults exposed from the versioned contract catalog."""

from typing import Final

from ._catalog import load_backend_defaults
from .base import ModelSettingDict

MOONSHOT_DEFAULT_MODEL: Final[str]
MOONSHOT_MODELS: Final[dict[str, ModelSettingDict]]
MOONSHOT_DEFAULT_MODEL, MOONSHOT_MODELS = load_backend_defaults("moonshot")
