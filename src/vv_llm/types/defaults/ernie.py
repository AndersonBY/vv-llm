"""Ernie defaults exposed from the versioned contract catalog."""

from typing import Final

from ._catalog import load_backend_defaults
from .base import ModelSettingDict

ERNIE_DEFAULT_MODEL: Final[str]
ERNIE_MODELS: Final[dict[str, ModelSettingDict]]
ERNIE_DEFAULT_MODEL, ERNIE_MODELS = load_backend_defaults("ernie")
