"""Baichuan defaults exposed from the versioned contract catalog."""

from typing import Final

from ._catalog import load_backend_defaults
from .base import ModelSettingDict

BAICHUAN_DEFAULT_MODEL: Final[str]
BAICHUAN_MODELS: Final[dict[str, ModelSettingDict]]
BAICHUAN_DEFAULT_MODEL, BAICHUAN_MODELS = load_backend_defaults("baichuan")
