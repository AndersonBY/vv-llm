"""Xiaomi defaults exposed from the versioned contract catalog."""

from typing import Final

from ._catalog import load_backend_defaults
from .base import ModelSettingDict

XIAOMI_DEFAULT_MODEL: Final[str]
XIAOMI_MODELS: Final[dict[str, ModelSettingDict]]
XIAOMI_DEFAULT_MODEL, XIAOMI_MODELS = load_backend_defaults("xiaomi")
