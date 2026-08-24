"""MiniMax defaults exposed from the versioned contract catalog."""

from typing import Final

from ._catalog import load_backend_defaults
from .base import ModelSettingDict

MINIMAX_DEFAULT_MODEL: Final[str]
MINIMAX_MODELS: Final[dict[str, ModelSettingDict]]
MINIMAX_DEFAULT_MODEL, MINIMAX_MODELS = load_backend_defaults("minimax")
