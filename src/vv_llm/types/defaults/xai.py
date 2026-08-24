"""xAI defaults exposed from the versioned contract catalog."""

from typing import Final

from ._catalog import load_backend_defaults
from .base import ModelSettingDict

XAI_DEFAULT_MODEL: Final[str]
XAI_MODELS: Final[dict[str, ModelSettingDict]]
XAI_DEFAULT_MODEL, XAI_MODELS = load_backend_defaults("xai")
